"""
ZX - ROTATION TRANSFORMATION ADAPTER (Image Subdivision of Z)
==============================================================

Learn visual transformation as an operator.
Like arithmetic learned addition structure, this learns rotation structure.

Task:
  Input:  Image (any image)
  Output: Rotated image (90°)
  Train:  Simple shapes (circles, squares, triangles)
  Test:   Complex images (faces, textures, photos)

Question: Can the network learn the TOPOLOGY of rotation and apply it
to images it never trained on?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List
import math

# Import dual-tetrahedral components
from Z_interface_coupling import DualTetrahedralNetwork, DualTetrahedralTrainer


# ============================================================================
# SIMPLE SHAPE GENERATOR
# ============================================================================

class ShapeGenerator:
    """
    Generate simple geometric shapes for training.

    Shapes: circles, squares, triangles, lines
    Variations: size, position, orientation
    """
    def __init__(self, img_size: int = 32):
        self.img_size = img_size

    def create_circle(self, center: Tuple[int, int], radius: int) -> np.ndarray:
        """Create a circle."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        y, x = np.ogrid[:self.img_size, :self.img_size]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = 1.0
        return img

    def create_square(self, center: Tuple[int, int], size: int) -> np.ndarray:
        """Create a square."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        half = size // 2
        x0, y0 = max(0, center[0] - half), max(0, center[1] - half)
        x1, y1 = min(self.img_size, center[0] + half), min(self.img_size, center[1] + half)
        img[y0:y1, x0:x1] = 1.0
        return img

    def create_triangle(self, center: Tuple[int, int], size: int) -> np.ndarray:
        """Create an upward-pointing triangle."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Triangle vertices (pointing up)
        cx, cy = center
        half = size // 2

        # Simple filled triangle using point-in-triangle test
        for y in range(self.img_size):
            for x in range(self.img_size):
                # Vertices: top, bottom-left, bottom-right
                v0 = (cx, cy - half)
                v1 = (cx - half, cy + half)
                v2 = (cx + half, cy + half)

                if self._point_in_triangle((x, y), v0, v1, v2):
                    img[y, x] = 1.0

        return img

    def create_line(self, start: Tuple[int, int], end: Tuple[int, int], thickness: int = 2) -> np.ndarray:
        """Create a line."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Bresenham's line algorithm (simplified with thickness)
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Draw thick point
            for ty in range(-thickness//2, thickness//2 + 1):
                for tx in range(-thickness//2, thickness//2 + 1):
                    py, px = y0 + ty, x0 + tx
                    if 0 <= py < self.img_size and 0 <= px < self.img_size:
                        img[py, px] = 1.0

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return img

    def _point_in_triangle(self, p: Tuple[int, int], v0: Tuple[int, int],
                          v1: Tuple[int, int], v2: Tuple[int, int]) -> bool:
        """Check if point p is inside triangle v0-v1-v2."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def generate_random_shapes(self, n_samples: int = 1000) -> List[np.ndarray]:
        """Generate random shapes for training."""
        images = []

        for _ in range(n_samples):
            shape_type = np.random.choice(['circle', 'square', 'triangle', 'line'])

            # Random position (keep away from edges)
            margin = 8
            center = (
                np.random.randint(margin, self.img_size - margin),
                np.random.randint(margin, self.img_size - margin)
            )

            # Random size
            size = np.random.randint(4, 10)

            if shape_type == 'circle':
                img = self.create_circle(center, size)
            elif shape_type == 'square':
                img = self.create_square(center, size * 2)
            elif shape_type == 'triangle':
                img = self.create_triangle(center, size * 2)
            else:  # line
                end = (
                    np.random.randint(margin, self.img_size - margin),
                    np.random.randint(margin, self.img_size - margin)
                )
                img = self.create_line(center, end, thickness=2)

            images.append(img)

        return images


# ============================================================================
# ROTATION UTILITIES
# ============================================================================

def rotate_image_90(img: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees clockwise."""
    return np.rot90(img, k=-1)  # k=-1 for clockwise

def rotate_image_torch(img_tensor: torch.Tensor, angle: int = 90) -> torch.Tensor:
    """
    Rotate batch of flattened images.

    Args:
        img_tensor: (batch, H*W) flattened images
        angle: Rotation angle (90, 180, 270)

    Returns:
        Rotated images (batch, H*W)
    """
    batch_size = img_tensor.size(0)
    H = W = int(math.sqrt(img_tensor.size(1)))

    # Reshape to images
    imgs = img_tensor.view(batch_size, H, W)

    # Rotate
    k = -angle // 90  # k for np.rot90 (negative for clockwise)
    rotated = torch.from_numpy(
        np.stack([np.rot90(img.numpy(), k=k) for img in imgs])
    ).float()

    # Flatten back
    return rotated.view(batch_size, -1)


# ============================================================================
# ROTATION DATASET
# ============================================================================

class RotationDataset(Dataset):
    """
    Dataset of (image, rotated_image) pairs.

    Trains the network to learn rotation as a transformation operator.
    """
    def __init__(self, images: List[np.ndarray], angle: int = 90):
        """
        Args:
            images: List of 2D numpy arrays (H, W)
            angle: Rotation angle in degrees
        """
        self.images = images
        self.angle = angle

        # Flatten images
        self.inputs = torch.tensor(
            [img.flatten() for img in images],
            dtype=torch.float32
        )

        # Rotate and flatten
        k = -angle // 90
        rotated = [np.rot90(img, k=k) for img in images]
        self.targets = torch.tensor(
            [img.flatten() for img in rotated],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def create_rotation_datasets(n_train: int = 1000, n_test: int = 200,
                            img_size: int = 32) -> Tuple[Dataset, Dataset]:
    """
    Create training and test datasets for rotation learning.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        img_size: Image size (square)

    Returns:
        train_dataset, test_dataset
    """
    print(f"🎨 Generating shape datasets...")
    generator = ShapeGenerator(img_size=img_size)

    train_images = generator.generate_random_shapes(n_train)
    test_images = generator.generate_random_shapes(n_test)

    train_dataset = RotationDataset(train_images, angle=90)
    test_dataset = RotationDataset(test_images, angle=90)

    print(f"✓ Train: {len(train_dataset)} image pairs")
    print(f"✓ Test: {len(test_dataset)} image pairs")
    print(f"  Image size: {img_size}×{img_size}")
    print(f"  Input dim: {img_size * img_size}")
    print(f"  Task: Learn 90° rotation")

    return train_dataset, test_dataset


def visualize_predictions(model: DualTetrahedralNetwork, test_images: List[np.ndarray],
                         n_examples: int = 5):
    """
    Visualize model predictions vs ground truth.

    Args:
        model: Trained dual-tetrahedral network
        test_images: List of test images
        n_examples: Number of examples to show
    """
    model.eval()
    img_size = int(math.sqrt(len(test_images[0].flatten())))

    print("\n" + "=" * 70)
    print("🎨 ROTATION PREDICTIONS")
    print("=" * 70)

    with torch.no_grad():
        for i in range(min(n_examples, len(test_images))):
            img = test_images[i]
            img_flat = torch.tensor(img.flatten(), dtype=torch.float32).unsqueeze(0)

            # Predict rotation
            pred_flat = model(img_flat)
            pred_img = pred_flat.view(img_size, img_size).numpy()

            # Ground truth rotation
            gt_img = rotate_image_90(img)

            # Calculate error
            error = np.abs(pred_img - gt_img).mean()

            print(f"\n📸 Example {i+1}:")
            print(f"  Mean pixel error: {error:.6f}")

            # Simple ASCII visualization (optional)
            if img_size <= 16:  # Only for small images
                print("  Original → Predicted → Ground Truth")
                for row in range(img_size):
                    orig_row = "".join(["█" if img[row, col] > 0.5 else " "
                                       for col in range(img_size)])
                    pred_row = "".join(["█" if pred_img[row, col] > 0.5 else " "
                                       for col in range(img_size)])
                    gt_row = "".join(["█" if gt_img[row, col] > 0.5 else " "
                                     for col in range(img_size)])
                    print(f"  {orig_row} → {pred_row} → {gt_row}")

    print("=" * 70)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Train dual-tetrahedral network on rotation transformation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("🔄 DUAL-TETRAHEDRAL ROTATION LEARNING")
    print("=" * 70)
    print("\n📚 CONCEPT:")
    print("  Like arithmetic learned addition STRUCTURE,")
    print("  can this learn rotation STRUCTURE?")
    print()
    print("  Train on: Simple shapes (circles, squares, triangles)")
    print("  Test on:  Same shapes + complex images (TODO)")
    print("  Goal:     Learn transformation operator, not memorize pixels")
    print()

    # ========================================================================
    # 1. CREATE DATASETS
    # ========================================================================
    img_size = 32
    train_dataset, test_dataset = create_rotation_datasets(
        n_train=1000,
        n_test=200,
        img_size=img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # ========================================================================
    # 2. CREATE MODEL
    # ========================================================================
    print("\n🏗️  Building dual-tetrahedral model...")
    input_dim = img_size * img_size  # Flattened image

    model = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=input_dim,  # Output is also flattened image
        latent_dim=64,
        coupling_strength=0.5,
        output_mode="weighted"
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Input: {img_size}×{img_size} image ({input_dim} dims)")
    print(f"  Output: Rotated image ({input_dim} dims)")
    print(f"  Linear net: Smooth rotation flow")
    print(f"  Nonlinear net: Preserve edges/boundaries")

    # ========================================================================
    # 3. TRAIN
    # ========================================================================
    print("\n⚡ Training...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = DualTetrahedralTrainer(model, optimizer, device)

    history = trainer.train(
        train_loader,
        test_loader,
        epochs=100,
        loss_fn=nn.MSELoss()
    )

    # ========================================================================
    # 4. VISUALIZE RESULTS
    # ========================================================================
    print("\n🎨 Generating visualizations...")
    generator = ShapeGenerator(img_size=img_size)
    test_images = generator.generate_random_shapes(n_samples=5)

    visualize_predictions(model, test_images, n_examples=5)

    # ========================================================================
    # 5. RESULTS
    # ========================================================================
    final_test_loss = history['test_loss'][-1]

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"\nFinal test loss: {final_test_loss:.6f}")

    if final_test_loss < 0.01:
        print("✅ EXCELLENT: Network learned rotation structure!")
    elif final_test_loss < 0.1:
        print("✓ GOOD: Network can rotate shapes reasonably well")
    else:
        print("⚠️  LEARNING: Network is still learning (may need more epochs)")

    print("\n💡 NEXT STEPS:")
    print("  1. Test on complex images (faces, textures)")
    print("  2. Try other angles (180°, 270°, 45°)")
    print("  3. Test composition (rotate twice = 180°?)")
    print("  4. Analyze what linear vs nonlinear networks learned")
    print("=" * 70)

    return model, trainer, history


if __name__ == "__main__":
    model, trainer, history = main()
    print("\n✓ Training complete!")


# ============================================================================
# SUMMARY
# ============================================================================

"""
ZX_rotation_adapter.py - Visual Transformation Learning

GOAL:
  Learn rotation as a transformation OPERATOR (like learning addition structure)

ARCHITECTURE:
  - Input: Flattened 32×32 image (1024 dims)
  - Dual tetrahedral networks (linear + nonlinear)
  - Output: Rotated image (1024 dims)

TRAINING:
  - Simple shapes: circles, squares, triangles, lines
  - 90° rotation only
  - ~1000 training samples

TEST GENERALIZATION:
  1. New shapes (same type, different positions)
  2. Complex images (faces, textures, photos)
  3. Other angles (180°, 270°, 45°)
  4. Composition (rotate twice = 180°?)

THE QUESTION:
  Did it learn the TOPOLOGY of rotation?
  Or did it just memorize pixel patterns?

  Arithmetic generalized 1000x.
  Can visual transformation generalize similarly?
"""
