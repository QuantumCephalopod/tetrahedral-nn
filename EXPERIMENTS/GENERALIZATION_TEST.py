"""
GENERALIZATION TEST - Does Rotation Structure Transfer?
========================================================

The moment of truth: Can the network rotate images it NEVER trained on?

Trained on: Simple shapes (circles, squares, triangles)
Test on: Complex patterns, textures, faces, anything we can find

This is the arithmetic generalization test for vision.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import math

# For generating test images
from PIL import Image, ImageDraw, ImageFont
import io


# ============================================================================
# COMPLEX IMAGE GENERATORS (When external images not available)
# ============================================================================

class ComplexImageGenerator:
    """
    Generate complex test images that are NOTHING like training shapes.
    """
    def __init__(self, img_size: int = 32):
        self.img_size = img_size

    def create_checkerboard(self, square_size: int = 4) -> np.ndarray:
        """Checkerboard pattern - very different from organic shapes."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for i in range(0, self.img_size, square_size):
            for j in range(0, self.img_size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    img[i:i+square_size, j:j+square_size] = 1.0
        return img

    def create_gradient(self, direction: str = 'horizontal') -> np.ndarray:
        """Smooth gradient - tests linear network's interpolation."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        if direction == 'horizontal':
            img = np.tile(np.linspace(0, 1, self.img_size), (self.img_size, 1))
        elif direction == 'vertical':
            img = np.tile(np.linspace(0, 1, self.img_size), (self.img_size, 1)).T
        elif direction == 'diagonal':
            x, y = np.meshgrid(np.linspace(0, 1, self.img_size),
                              np.linspace(0, 1, self.img_size))
            img = (x + y) / 2
        return img

    def create_concentric_circles(self) -> np.ndarray:
        """Concentric circles - structured but complex."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        center = self.img_size // 2
        y, x = np.ogrid[:self.img_size, :self.img_size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)

        # Multiple rings
        for radius in range(2, self.img_size // 2, 3):
            mask = (dist >= radius - 1) & (dist <= radius + 1)
            img[mask] = 1.0
        return img

    def create_grid_lines(self, spacing: int = 8) -> np.ndarray:
        """Grid of lines - sharp edges everywhere."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Horizontal lines
        for i in range(0, self.img_size, spacing):
            img[i, :] = 1.0

        # Vertical lines
        for j in range(0, self.img_size, spacing):
            img[:, j] = 1.0

        return img

    def create_random_noise(self, density: float = 0.3) -> np.ndarray:
        """Random noise pattern - chaos test."""
        img = (np.random.random((self.img_size, self.img_size)) < density).astype(np.float32)
        return img

    def create_letter(self, letter: str = 'A') -> np.ndarray:
        """
        Create a letter/text - completely different from geometric shapes.
        Tests if it learned general rotation or just shape rotation.
        """
        # Create PIL image
        pil_img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(pil_img)

        # Try to use a font (fallback to default if not available)
        try:
            # Try to load a font (size depends on image size)
            font_size = self.img_size // 2
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                     font_size)
        except:
            font = ImageFont.load_default()

        # Draw text centered
        text_bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((self.img_size - text_width) // 2,
                   (self.img_size - text_height) // 2)

        draw.text(position, letter, fill=255, font=font)

        # Convert to numpy
        img = np.array(pil_img, dtype=np.float32) / 255.0
        return img

    def create_spiral(self) -> np.ndarray:
        """Spiral pattern - complex curved structure."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        center = self.img_size / 2

        for t in np.linspace(0, 4 * np.pi, 500):
            r = t * 2
            x = int(center + r * np.cos(t))
            y = int(center + r * np.sin(t))

            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                # Draw thick point
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.img_size and 0 <= ny < self.img_size:
                            img[ny, nx] = 1.0

        return img


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_rotation_test(model, test_images: List[Tuple[str, np.ndarray]],
                            img_size: int = 32, save_path: str = None):
    """
    Visualize rotation predictions on complex images.

    Args:
        model: Trained rotation model
        test_images: List of (name, image) tuples
        img_size: Image size
        save_path: Optional path to save figure
    """
    model.eval()

    n_images = len(test_images)
    fig, axes = plt.subplots(n_images, 4, figsize=(12, 3 * n_images))

    if n_images == 1:
        axes = axes.reshape(1, -1)

    print("\n" + "=" * 70)
    print("🎨 GENERALIZATION TEST - Complex Images")
    print("=" * 70)

    with torch.no_grad():
        for i, (name, img) in enumerate(test_images):
            # Prepare input
            img_flat = torch.tensor(img.flatten(), dtype=torch.float32).unsqueeze(0)

            # Predict rotation
            pred_flat = model(img_flat)
            pred_img = pred_flat.view(img_size, img_size).numpy()

            # Ground truth rotation
            gt_img = np.rot90(img, k=-1)  # 90° clockwise

            # Calculate errors
            pixel_error = np.abs(pred_img - gt_img).mean()
            max_error = np.abs(pred_img - gt_img).max()

            # Visual similarity (structural)
            # Simple metric: correlation coefficient
            correlation = np.corrcoef(pred_img.flatten(), gt_img.flatten())[0, 1]

            print(f"\n📸 {name}:")
            print(f"  Mean pixel error: {pixel_error:.6f}")
            print(f"  Max pixel error:  {max_error:.6f}")
            print(f"  Correlation:      {correlation:.6f}")

            # Verdict
            if pixel_error < 0.01:
                verdict = "✨ PERFECT - Learned structure!"
            elif pixel_error < 0.05:
                verdict = "✅ EXCELLENT - Strong generalization"
            elif pixel_error < 0.1:
                verdict = "✓ GOOD - Reasonable transfer"
            elif pixel_error < 0.2:
                verdict = "○ OK - Partial transfer"
            else:
                verdict = "△ WEAK - Limited generalization"

            print(f"  Verdict: {verdict}")

            # Plot
            axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Original\n{name}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Predicted\nError: {pixel_error:.4f}')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Ground Truth\n(90° rotation)')
            axes[i, 2].axis('off')

            # Error map
            error_map = np.abs(pred_img - gt_img)
            axes[i, 3].imshow(error_map, cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Error Map\nMax: {max_error:.4f}')
            axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {save_path}")

    plt.show()

    print("=" * 70)


# ============================================================================
# ASCII VISUALIZATION (For terminal display)
# ============================================================================

def show_ascii_comparison(original: np.ndarray, predicted: np.ndarray,
                         ground_truth: np.ndarray, name: str):
    """
    Show ASCII art comparison for smaller images.
    """
    if original.shape[0] > 32:
        return  # Too large for ASCII

    print(f"\n🎨 {name} (ASCII visualization):")
    print("  Original → Predicted → Ground Truth")
    print("  " + "─" * (original.shape[1] * 3 + 6))

    for row in range(original.shape[0]):
        orig_row = "".join(["█" if original[row, col] > 0.5 else "·"
                           for col in range(original.shape[1])])
        pred_row = "".join(["█" if predicted[row, col] > 0.5 else "·"
                           for col in range(predicted.shape[1])])
        gt_row = "".join(["█" if ground_truth[row, col] > 0.5 else "·"
                         for col in range(ground_truth.shape[1])])

        print(f"  {orig_row} → {pred_row} → {gt_row}")


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_generalization(model, img_size: int = 32, show_plots: bool = True):
    """
    Comprehensive generalization test on complex images.

    Args:
        model: Trained rotation model
        img_size: Image size
        show_plots: Whether to show matplotlib plots
    """
    print("\n" + "=" * 70)
    print("🔬 GENERALIZATION TEST")
    print("=" * 70)
    print("\nQuestion: Did the network learn ROTATION STRUCTURE?")
    print("  or just memorize SHAPE PATTERNS?")
    print()
    print("Training: Simple shapes (circles, squares, triangles)")
    print("Testing:  Complex patterns NEVER seen before")
    print()

    # Generate complex test images
    generator = ComplexImageGenerator(img_size=img_size)

    test_images = [
        ("Checkerboard", generator.create_checkerboard(square_size=4)),
        ("Horizontal Gradient", generator.create_gradient('horizontal')),
        ("Diagonal Gradient", generator.create_gradient('diagonal')),
        ("Concentric Circles", generator.create_concentric_circles()),
        ("Grid Lines", generator.create_grid_lines(spacing=6)),
        ("Random Noise", generator.create_random_noise(density=0.3)),
        ("Spiral", generator.create_spiral()),
    ]

    # Try to add letter if font available
    try:
        letter_img = generator.create_letter('A')
        if letter_img.sum() > 0:  # Check if letter was actually drawn
            test_images.append(("Letter 'A'", letter_img))
    except:
        pass

    # Visualize
    if show_plots:
        visualize_rotation_test(model, test_images, img_size=img_size)
    else:
        # Just print results without plots
        model.eval()
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)

        with torch.no_grad():
            for name, img in test_images:
                img_flat = torch.tensor(img.flatten(), dtype=torch.float32).unsqueeze(0)
                pred_flat = model(img_flat)
                pred_img = pred_flat.view(img_size, img_size).numpy()
                gt_img = np.rot90(img, k=-1)

                error = np.abs(pred_img - gt_img).mean()

                print(f"{name:25s} → Error: {error:.6f}")

    print("\n💡 INTERPRETATION:")
    print("  • Error < 0.05: Learned STRUCTURE (like arithmetic!)")
    print("  • Error 0.05-0.15: Partial generalization")
    print("  • Error > 0.15: Memorized patterns, not structure")
    print("=" * 70)


# ============================================================================
# COLAB-READY EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("ℹ️  Import this module and run:")
    print("    test_generalization(trained_model)")
    print()
    print("  Or for plots:")
    print("    test_generalization(trained_model, show_plots=True)")


# ============================================================================
# SUMMARY
# ============================================================================

"""
GENERALIZATION_TEST.py - The Moment of Truth

This is the equivalent of testing arithmetic on millions after training on [-9, 9].

We trained on simple shapes. Can it rotate:
  ✓ Checkerboards
  ✓ Gradients
  ✓ Grids
  ✓ Letters
  ✓ Spirals
  ✓ Noise

If YES → It learned rotation TOPOLOGY (SO(2) group structure)
If NO  → It memorized shape-specific patterns

The answer will tell us if geometric deep learning can discover
visual OPERATIONS (not just visual content).
"""
