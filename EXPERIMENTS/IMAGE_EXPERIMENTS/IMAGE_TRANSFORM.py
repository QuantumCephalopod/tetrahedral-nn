"""
IMAGE_TRANSFORM - Dual-Tetrahedral Image-to-Image Learning
===========================================================

Minimal-sample image transformation learning.
Testing if dual tetrahedra can learn transformation STRUCTURE from only 7 pairs.

Experiment:
  - 7 image pairs: [input] → [output] with consistent transformation
  - Exhaustive augmentation: 4 rotations × 4 flip states = 16 variations/pair
  - Train: 6 pairs (96 samples)
  - Test: 1 held-out pair (16 samples)
  - Full resolution: 512×512×3 RGB

Question: Can the network learn the relational structure of the transformation
from minimal data, or will it overfit to pixel patterns?

Philosophy: Raw signal, no preprocessing, trust self-organization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Import dual-tetrahedral components
from Z_interface_coupling import DualTetrahedralNetwork, DualTetrahedralTrainer


# ============================================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================================

class ImagePairLoader:
    """
    Load image pairs from folders with automatic format conversion and resizing.

    Handles: webp, png, jpg, jpeg
    Output: 512×512 RGB normalized to [0, 1]
    """
    def __init__(self, input_folder: str, output_folder: str, target_size: int = 512):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.target_size = target_size

    def load_and_prepare(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and prepare an image pair.

        Args:
            filename: Base filename (will search for various extensions)

        Returns:
            (input_tensor, output_tensor) - both (3, 512, 512) normalized to [0,1]
        """
        # Find input file (try various extensions)
        input_path = self._find_file(self.input_folder, filename)
        output_path = self._find_file(self.output_folder, filename)

        if input_path is None or output_path is None:
            raise FileNotFoundError(f"Could not find pair for {filename}")

        # Load and process
        input_img = self._load_image(input_path)
        output_img = self._load_image(output_path)

        return input_img, output_img

    def _find_file(self, folder: Path, base_name: str) -> Optional[Path]:
        """Find file with any common image extension."""
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP']

        # Try exact match first
        for ext in extensions:
            path = folder / f"{base_name}{ext}"
            if path.exists():
                return path

        # Try without extension (maybe it's already included)
        path = folder / base_name
        if path.exists():
            return path

        return None

    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load image and convert to tensor.

        Returns:
            Tensor of shape (3, 512, 512) normalized to [0, 1]
        """
        # Load with PIL (handles all formats)
        img = Image.open(path).convert('RGB')

        # Resize to target size
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)

        # Convert to tensor: (H, W, C) -> (C, H, W), normalized to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        return img_tensor

    def load_all_pairs(self, filenames: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load multiple image pairs.

        Args:
            filenames: List of base filenames

        Returns:
            List of (input, output) tensor pairs
        """
        pairs = []
        print(f"📂 Loading {len(filenames)} image pairs...")

        for filename in filenames:
            try:
                pair = self.load_and_prepare(filename)
                pairs.append(pair)
                print(f"  ✓ Loaded: {filename}")
            except Exception as e:
                print(f"  ✗ Failed to load {filename}: {e}")
                raise

        print(f"✓ Loaded {len(pairs)} pairs successfully\n")
        return pairs


# ============================================================================
# EXHAUSTIVE AUGMENTATION
# ============================================================================

def augment_image_exhaustive(img: torch.Tensor) -> List[torch.Tensor]:
    """
    Create all 16 exhaustive augmentations of an image.

    4 rotations (0°, 90°, 180°, 270°) × 4 flip states (none, H, V, both) = 16

    Args:
        img: Image tensor (C, H, W)

    Returns:
        List of 16 augmented tensors
    """
    augmentations = []

    # 4 rotation states
    for k in range(4):  # k=0: 0°, k=1: 90° CCW, k=2: 180°, k=3: 270° CCW
        rotated = torch.rot90(img, k=k, dims=[1, 2])

        # 4 flip states per rotation
        augmentations.append(rotated)                          # No flip
        augmentations.append(torch.flip(rotated, dims=[2]))    # H-flip
        augmentations.append(torch.flip(rotated, dims=[1]))    # V-flip
        augmentations.append(torch.flip(rotated, dims=[1, 2])) # Both flips

    return augmentations


# ============================================================================
# DATASET
# ============================================================================

class ImageTransformDataset(Dataset):
    """
    Dataset for image-to-image transformation learning.

    Takes image pairs and applies exhaustive augmentation.
    """
    def __init__(self, image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                 apply_augmentation: bool = True):
        """
        Args:
            image_pairs: List of (input, output) tensor pairs
            apply_augmentation: If True, create 16 augmentations per pair
        """
        self.inputs = []
        self.outputs = []

        for input_img, output_img in image_pairs:
            if apply_augmentation:
                # Exhaustive augmentation: 16 variations
                input_augs = augment_image_exhaustive(input_img)
                output_augs = augment_image_exhaustive(output_img)

                for inp, out in zip(input_augs, output_augs):
                    self.inputs.append(inp)
                    self.outputs.append(out)
            else:
                # No augmentation
                self.inputs.append(input_img)
                self.outputs.append(output_img)

        print(f"  Dataset size: {len(self.inputs)} samples")
        if apply_augmentation:
            print(f"  ({len(image_pairs)} pairs × 16 augmentations)")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Flatten images: (3, 512, 512) -> (786432,)
        input_flat = self.inputs[idx].reshape(-1)
        output_flat = self.outputs[idx].reshape(-1)
        return input_flat, output_flat


# ============================================================================
# TRAINING SETUP
# ============================================================================

def create_datasets(input_folder: str, output_folder: str,
                   filenames: List[str], test_idx: int = 6) -> Tuple[Dataset, Dataset]:
    """
    Create train and test datasets with exhaustive augmentation.

    Args:
        input_folder: Path to input images
        output_folder: Path to output images
        filenames: List of 7 base filenames
        test_idx: Index of pair to hold out for testing (0-6)

    Returns:
        train_dataset, test_dataset
    """
    print("=" * 70)
    print("📊 CREATING DATASETS")
    print("=" * 70)

    # Load all pairs
    loader = ImagePairLoader(input_folder, output_folder, target_size=512)
    all_pairs = loader.load_all_pairs(filenames)

    # Split into train/test
    train_pairs = [pair for i, pair in enumerate(all_pairs) if i != test_idx]
    test_pairs = [all_pairs[test_idx]]

    print(f"📦 Creating datasets with exhaustive augmentation...")
    print(f"\n🎯 Train set:")
    print(f"  Base pairs: {len(train_pairs)}")
    train_dataset = ImageTransformDataset(train_pairs, apply_augmentation=True)

    print(f"\n🎯 Test set:")
    print(f"  Base pairs: {len(test_pairs)} (held-out pair {test_idx})")
    test_dataset = ImageTransformDataset(test_pairs, apply_augmentation=True)

    print("\n" + "=" * 70)
    return train_dataset, test_dataset


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model: DualTetrahedralNetwork, test_dataset: Dataset,
                         device: str = 'cpu', n_examples: int = 4):
    """
    Visualize model predictions vs ground truth.

    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on
        n_examples: Number of examples to show
    """
    model.eval()

    print("\n" + "=" * 70)
    print("🎨 VISUALIZATION")
    print("=" * 70)

    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(min(n_examples, len(test_dataset))):
            input_flat, target_flat = test_dataset[i]
            input_flat = input_flat.unsqueeze(0).to(device)

            # Predict
            pred_flat = model(input_flat).cpu()

            # Reshape to images
            input_img = input_flat.cpu().view(3, 512, 512).permute(1, 2, 0).numpy()
            pred_img = pred_flat.view(3, 512, 512).permute(1, 2, 0).numpy()
            target_img = target_flat.view(3, 512, 512).permute(1, 2, 0).numpy()

            # Clip to [0, 1]
            pred_img = np.clip(pred_img, 0, 1)

            # Calculate error
            mse = np.mean((pred_img - target_img) ** 2)

            # Plot
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title(f"Input {i+1}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f"Predicted (MSE: {mse:.6f})")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(target_img)
            axes[i, 2].set_title(f"Target {i+1}")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'predictions.png'")
    plt.show()


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main(input_folder: str, output_folder: str, filenames: List[str],
         test_idx: int = 6, epochs: int = 500, latent_dim: int = 4096,
         batch_size: int = 8, lr: float = 0.0001):
    """
    Complete training pipeline for image transformation learning.

    Args:
        input_folder: Path to input images folder
        output_folder: Path to output images folder
        filenames: List of 7 base filenames (without extensions)
        test_idx: Which pair to hold out for testing (0-6)
        epochs: Number of training epochs
        latent_dim: Vertex dimension (4096 recommended)
        batch_size: Training batch size
        lr: Learning rate
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 70)
    print("🌀 DUAL-TETRAHEDRAL IMAGE TRANSFORMATION")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Image size: 512×512×3 = 786,432 dimensions")
    print(f"Latent dimension: {latent_dim}")
    print(f"Training setup: 6 pairs (96 samples) → 1 held-out pair (16 samples)")
    print("\n" + "=" * 70 + "\n")

    # ========================================================================
    # 1. CREATE DATASETS
    # ========================================================================
    train_dataset, test_dataset = create_datasets(
        input_folder, output_folder, filenames, test_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ========================================================================
    # 2. CREATE MODEL
    # ========================================================================
    print("\n🏗️  BUILDING MODEL")
    print("=" * 70)

    model = DualTetrahedralNetwork(
        input_dim=786432,      # 512×512×3 flattened
        output_dim=786432,     # Same size output
        latent_dim=latent_dim,
        coupling_strength=0.5,
        output_mode="weighted"
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Dual-tetrahedral network created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Linear tetrahedron: 4 vertices, smooth transformations")
    print(f"  Nonlinear tetrahedron: 4 vertices, boundary detection")
    print(f"  Inter-face coupling: 8 attention modules")
    print(f"  Input projection: 786,432 → {latent_dim} × 4 per network")
    print(f"  Output projection: {latent_dim} × 8 → 786,432")

    # ========================================================================
    # 3. TRAIN
    # ========================================================================
    print("\n" + "=" * 70)
    print("⚡ TRAINING")
    print("=" * 70)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = DualTetrahedralTrainer(model, optimizer, device)

    print(f"Optimizer: Adam (lr={lr})")
    print(f"Loss: MSE (pixel-wise reconstruction)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("\nStarting training...\n")

    history = trainer.train(
        train_loader,
        test_loader,
        epochs=epochs,
        loss_fn=nn.MSELoss()
    )

    # ========================================================================
    # 4. EVALUATE
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 EVALUATION")
    print("=" * 70)

    final_train_loss = history['train_loss'][-1]
    final_test_loss = history['test_loss'][-1]

    print(f"\nFinal train loss: {final_train_loss:.6f}")
    print(f"Final test loss:  {final_test_loss:.6f}")

    if final_test_loss < 0.001:
        print("\n✅ EXCELLENT: Network learned transformation structure!")
    elif final_test_loss < 0.01:
        print("\n✓ GOOD: Network can generalize reasonably well")
    else:
        print("\n⚠️  LEARNING: Network may need more training or tuning")

    # ========================================================================
    # 5. VISUALIZE
    # ========================================================================
    visualize_predictions(model, test_dataset, device, n_examples=4)

    # ========================================================================
    # 6. SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("💡 EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\n📈 Key Questions:")
    print("  1. Did it generalize to the held-out image?")
    print("  2. Did it learn transformation structure or memorize pixels?")
    print("  3. How do linear vs nonlinear networks contribute?")
    print("\n🔬 Next Explorations:")
    print("  • Test with different transformations")
    print("  • Analyze network contributions (get_network_contributions)")
    print("  • Try different loss functions (relational/consensus-based)")
    print("  • Visualize attention patterns in faces/edges")
    print("=" * 70 + "\n")

    return model, trainer, history


# ============================================================================
# COLAB QUICKSTART
# ============================================================================

def colab_quickstart():
    """
    Quick start for Google Colab.

    Usage:
        1. Mount Google Drive
        2. Update paths below to your image folders
        3. Run this function
    """
    print("🚀 COLAB QUICKSTART")
    print("=" * 70)
    print("\n⚠️  SETUP REQUIRED:")
    print("  1. Mount Google Drive")
    print("  2. Update folder paths below")
    print("  3. Provide list of 7 base filenames")
    print("\n" + "=" * 70 + "\n")

    # ========================================================================
    # CUSTOMIZE THESE PATHS
    # ========================================================================

    # Example paths (update to your actual paths)
    input_folder = "/content/drive/MyDrive/image_transform/input"
    output_folder = "/content/drive/MyDrive/image_transform/output"

    # Your 7 base filenames (without extensions)
    # The loader will automatically find .jpg, .png, .webp, etc.
    filenames = [
        "image_01",
        "image_02",
        "image_03",
        "image_04",
        "image_05",
        "image_06",
        "image_07"
    ]

    # ========================================================================
    # RUN
    # ========================================================================

    print("📂 Using folders:")
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"\n📋 Files: {filenames}")
    print(f"\n🎯 Test pair: {filenames[6]} (held out)")
    print("\n" + "=" * 70 + "\n")

    # Run training
    model, trainer, history = main(
        input_folder=input_folder,
        output_folder=output_folder,
        filenames=filenames,
        test_idx=6,          # Hold out last pair
        epochs=500,          # Adjust as needed
        latent_dim=4096,     # 4096 recommended
        batch_size=8,        # Depends on GPU memory
        lr=0.0001           # Conservative for stability
    )

    return model, trainer, history


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    For Colab: Update paths in colab_quickstart() and run:
        model, trainer, history = colab_quickstart()

    For custom use:
        model, trainer, history = main(
            input_folder="path/to/inputs",
            output_folder="path/to/outputs",
            filenames=["img1", "img2", ...],
            test_idx=6,
            epochs=500
        )
    """
    print("\n" + "=" * 70)
    print("IMAGE_TRANSFORM.py")
    print("=" * 70)
    print("\n📖 This file contains:")
    print("  • ImagePairLoader - Load and preprocess image pairs")
    print("  • Exhaustive augmentation - 16 variations per pair")
    print("  • ImageTransformDataset - PyTorch dataset")
    print("  • Training pipeline - Complete training loop")
    print("  • Visualization - Compare predictions vs targets")
    print("\n🚀 To use in Colab:")
    print("  1. Update paths in colab_quickstart()")
    print("  2. Run: model, trainer, history = colab_quickstart()")
    print("\n💡 Philosophy:")
    print("  Raw 512×512 RGB → 8 vertices → Self-organization")
    print("  No encoding/decoding, clean signal, trust the structure")
    print("=" * 70 + "\n")


# ============================================================================
# SUMMARY
# ============================================================================

"""
IMAGE_TRANSFORM.py - Minimal-Sample Image Transformation Learning

EXPERIMENT DESIGN:
  • 7 image pairs with consistent A→B transformation
  • Exhaustive augmentation: 4 rotations × 4 flip states = 16/pair
  • Train: 6 pairs (96 samples)
  • Test: 1 held-out pair (16 samples)
  • Full resolution: 512×512×3 = 786,432 dimensions

ARCHITECTURE:
  • Direct input: No encoding, raw pixels → 8 vertices
  • Dual tetrahedra: Linear (smooth) + Nonlinear (boundaries)
  • Face-to-face coupling: Pattern-level communication
  • Direct output: 8 vertices → raw pixels
  • Latent dimension: 4096 per vertex (recommended)

PHILOSOPHY:
  • Self-organization through geometric constraints
  • Clean signal preservation (no lossy encoding)
  • Trust the tetrahedral structure to handle dimensionality
  • Test: Does it learn transformation STRUCTURE or memorize pixels?

KEY QUESTION:
  Can dual tetrahedra learn relational transformation structure
  from only 6 training pairs and generalize to a 7th unseen image?

This is the image equivalent of arithmetic generalization:
  Trained on simple examples, extrapolates to new instances.

Next: Explore relational/consensus-based loss functions.
"""
