"""
CONSENSUS_ONLY - Pure Consensus Training (No MSE Comparison)
=============================================================

Single-focus experiment: Train with consensus loss ONLY.
No comparisons, no MSE, no distractions.

Philosophy:
  "Reality emerges from perspective agreement, not ground truth matching."

Author: Philipp Remy Bartholomäus
Date: November 2025
"""

# ============================================================================
# COLAB CELL: Consensus Training Only
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import random

# Assumes DualTetrahedralNetwork is already imported or available
# If not: from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork


# ----------------------------------------------------------------------------
# Image Loading
# ----------------------------------------------------------------------------

def load_image_pairs(input_folder, output_folder, img_size=128):
    """Load all image pairs from folders."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    input_files = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.webp')))
    output_files = sorted(list(output_path.glob('*.png')) + list(output_path.glob('*.jpg')) + list(output_path.glob('*.webp')))

    pairs = []
    print(f"📂 Loading {len(input_files)} image pairs...")

    for inp, out in zip(input_files, output_files):
        img_in = Image.open(inp).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
        img_out = Image.open(out).convert('RGB').resize((img_size, img_size), Image.LANCZOS)

        tensor_in = torch.from_numpy(np.array(img_in, dtype=np.float32) / 255.0).permute(2, 0, 1)
        tensor_out = torch.from_numpy(np.array(img_out, dtype=np.float32) / 255.0).permute(2, 0, 1)

        pairs.append((tensor_in, tensor_out))

    print(f"✓ Loaded {len(pairs)} pairs\n")
    return pairs


# ----------------------------------------------------------------------------
# Multi-Representation Transforms
# ----------------------------------------------------------------------------

def create_edge_version(img: torch.Tensor) -> torch.Tensor:
    """Create edge-detected version using Canny."""
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_3ch = np.stack([edges, edges, edges], axis=2) / 255.0
    return torch.from_numpy(edges_3ch).permute(2, 0, 1).float()


def create_grayscale_version(img: torch.Tensor) -> torch.Tensor:
    """Create grayscale version."""
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_3ch = np.stack([gray, gray, gray], axis=2) / 255.0
    return torch.from_numpy(gray_3ch).permute(2, 0, 1).float()


def create_dithered_version(img: torch.Tensor) -> torch.Tensor:
    """Create simple binary dithered version."""
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    dithered = (gray > 128).astype(np.uint8) * 255
    dithered_3ch = np.stack([dithered, dithered, dithered], axis=2) / 255.0
    return torch.from_numpy(dithered_3ch).permute(2, 0, 1).float()


# ----------------------------------------------------------------------------
# Multi-Representation Dataset
# ----------------------------------------------------------------------------

class MultiRepDataset(Dataset):
    """
    Dataset with 4 representations per sample:
      0. Original RGB
      1. Edge-detected
      2. Grayscale
      3. Dithered
    """

    def __init__(self, image_pairs, apply_augmentation=True):
        self.base_pairs = []

        # Simple augmentation (4 rotations × 2 flips = 8)
        for input_img, output_img in image_pairs:
            if apply_augmentation:
                augs_in = self._augment(input_img)
                augs_out = self._augment(output_img)
                for aug_in, aug_out in zip(augs_in, augs_out):
                    self.base_pairs.append((aug_in, aug_out))
            else:
                self.base_pairs.append((input_img, output_img))

        self.n_base = len(self.base_pairs)
        print(f"  Base pairs: {self.n_base}, Total samples: {self.n_base * 4}")

    def _augment(self, img):
        """4 rotations × 2 flips = 8 augmentations."""
        augs = []
        for k in range(4):
            rotated = torch.rot90(img, k=k, dims=[1, 2])
            augs.append(rotated)
            augs.append(torch.flip(rotated, dims=[2]))
        return augs

    def __len__(self):
        return self.n_base * 4

    def __getitem__(self, idx):
        base_idx = idx // 4
        rep_type = idx % 4
        input_img, output_img = self.base_pairs[base_idx]

        # Create representation
        if rep_type == 0:
            inp, out = input_img, output_img
        elif rep_type == 1:
            inp = create_edge_version(input_img)
            out = create_edge_version(output_img)
        elif rep_type == 2:
            inp = create_grayscale_version(input_img)
            out = create_grayscale_version(output_img)
        else:
            inp = create_dithered_version(input_img)
            out = create_dithered_version(output_img)

        return inp.reshape(-1), out.reshape(-1), base_idx, rep_type


# ----------------------------------------------------------------------------
# Hybrid Batch Sampler
# ----------------------------------------------------------------------------

class HybridBatchSampler(Sampler):
    """
    Mix structured and random batches.

    structured_ratio=1.0: All batches have same base pair, 4 reps (teaching)
    structured_ratio=0.5: 50% structured, 50% random (balanced)
    structured_ratio=0.0: Fully random batches (real-world experience)
    """

    def __init__(self, dataset, structured_ratio=0.5, shuffle=True):
        self.n_base = dataset.n_base
        self.total_samples = len(dataset)
        self.structured_ratio = structured_ratio
        self.shuffle = shuffle

    def __iter__(self):
        n_structured = int(self.n_base * self.structured_ratio)
        n_random = self.n_base - n_structured

        batches = []
        base_indices = list(range(self.n_base))
        if self.shuffle:
            random.shuffle(base_indices)

        # Structured batches (same base pair, all 4 reps)
        for i in range(n_structured):
            base_idx = base_indices[i]
            batch = [base_idx * 4 + rep for rep in range(4)]
            batches.append(batch)

        # Random batches (any 4 samples)
        all_indices = list(range(self.total_samples))
        if self.shuffle:
            random.shuffle(all_indices)

        for i in range(n_random):
            start = i * 4
            batch = all_indices[start:start+4]
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.n_base


# ----------------------------------------------------------------------------
# Consensus Training Loop
# ----------------------------------------------------------------------------

def train_consensus(model, loader, optimizer, device, latent_dim=128,
                    consensus_weight=0.6, target_weight=0.4):
    """
    Train with consensus loss.

    Loss = 0.6 * agreement(Linear, Nonlinear) + 0.4 * target_guidance
    """
    model.train()
    total_loss = 0.0
    total_consensus = 0.0
    total_target = 0.0

    for batch_data in loader:
        batch_x, batch_y, _, _ = batch_data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        bs = batch_x.size(0)

        optimizer.zero_grad()

        # Forward through both networks
        lin_v, lin_f = model.linear_net(batch_x, return_faces=True)
        non_v, non_f = model.nonlinear_net(batch_x, return_faces=True)

        # Couple faces
        coupled_l = [model.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i]) for i in range(4)]
        coupled_n = [model.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i]) for i in range(4)]

        lin_v = model.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
        non_v = model.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

        # Get separate network outputs
        lin_vertices_only = torch.cat([
            lin_v.reshape(bs, -1),
            torch.zeros(bs, latent_dim * 4, device=device)
        ], dim=-1)

        non_vertices_only = torch.cat([
            torch.zeros(bs, latent_dim * 4, device=device),
            non_v.reshape(bs, -1)
        ], dim=-1)

        linear_output = model.output_projection(lin_vertices_only)
        nonlinear_output = model.output_projection(non_vertices_only)

        # CONSENSUS LOSS: Networks must agree
        consensus_loss = F.mse_loss(linear_output, nonlinear_output)

        # TARGET GUIDANCE: Both orbit around target
        target_loss = (F.mse_loss(linear_output, batch_y) +
                      F.mse_loss(nonlinear_output, batch_y)) / 2

        # Combined loss
        loss = consensus_weight * consensus_loss + target_weight * target_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_consensus += consensus_loss.item()
        total_target += target_loss.item()

    n = len(loader)
    return total_loss / n, total_consensus / n, total_target / n


def evaluate(model, loader, device):
    """Simple MSE evaluation."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_data in loader:
            batch_x, batch_y, _, _ = batch_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            total_loss += F.mse_loss(output, batch_y).item()

    return total_loss / len(loader)


# ----------------------------------------------------------------------------
# Main Training Function
# ----------------------------------------------------------------------------

def run_consensus_training(input_folder, output_folder,
                          img_size=128, latent_dim=128,
                          epochs=100, test_idx=6,
                          structured_ratio=0.5,
                          consensus_weight=0.6, target_weight=0.4,
                          device=None):
    """
    Run consensus training.

    Args:
        structured_ratio: 0.0-1.0, ratio of structured batches
                         1.0 = all structured (same pair, 4 reps)
                         0.0 = all random

    Returns:
        model, history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("🌀 CONSENSUS TRAINING")
    print("="*70)
    print(f"Image size: {img_size}×{img_size}")
    print(f"Latent dim: {latent_dim}")
    print(f"Structured ratio: {structured_ratio}")
    print(f"Consensus weight: {consensus_weight}")
    print(f"Target weight: {target_weight}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    # Load data
    all_pairs = load_image_pairs(input_folder, output_folder, img_size)

    if len(all_pairs) == 0:
        raise ValueError("No image pairs found! Check your input/output folder paths.")

    # Split train/test
    if test_idx >= len(all_pairs):
        print(f"⚠️  Warning: test_idx={test_idx} is out of range (only {len(all_pairs)} pairs)")
        print(f"   Using last pair as test set instead\n")
        test_idx = len(all_pairs) - 1

    train_pairs = [p for i, p in enumerate(all_pairs) if i != test_idx]
    test_pairs = [all_pairs[test_idx]]

    # Create multi-rep datasets
    print("📦 Creating datasets...")
    train_dataset = MultiRepDataset(train_pairs, apply_augmentation=True)
    test_dataset = MultiRepDataset(test_pairs, apply_augmentation=True)

    # Create hybrid batch samplers
    train_sampler = HybridBatchSampler(train_dataset, structured_ratio=structured_ratio)
    test_sampler = HybridBatchSampler(test_dataset, structured_ratio=0.0, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}\n")

    # Create model
    input_dim = img_size * img_size * 3
    model = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=latent_dim,
        coupling_strength=0.5,
        output_mode="weighted"
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train
    history = {'train': [], 'test': [], 'consensus': [], 'target': []}

    print("Training...")
    print("="*70)

    for epoch in range(epochs):
        train_loss, consensus, target = train_consensus(
            model, train_loader, optimizer, device,
            latent_dim=latent_dim,
            consensus_weight=consensus_weight,
            target_weight=target_weight
        )

        test_loss = evaluate(model, test_loader, device)

        history['train'].append(train_loss)
        history['test'].append(test_loss)
        history['consensus'].append(consensus)
        history['target'].append(target)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Test: {test_loss:.6f} | "
                  f"Consensus: {consensus:.6f} | "
                  f"Target: {target:.6f}")

    print("\n✅ Training complete!\n")

    # Visualize
    visualize_results(model, train_dataset, history, img_size, device)

    return model, history


def visualize_results(model, train_dataset, history, img_size, device):
    """Visualize training results."""
    model.eval()

    # Get a sample from base pairs
    sample_x, sample_y = train_dataset.base_pairs[0]
    sample_x_tensor = sample_x.reshape(1, -1).to(device)

    with torch.no_grad():
        output = model(sample_x_tensor).cpu()

    # Reshape to images
    def to_img(t):
        return t.reshape(3, img_size, img_size).permute(1, 2, 0).numpy().clip(0, 1)

    input_img = to_img(sample_x)
    target_img = to_img(sample_y)
    output_img = to_img(output)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_img)
    axes[0, 1].set_title('Target', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(output_img)
    axes[1, 0].set_title('Consensus Output', fontsize=14, fontweight='bold', color='blue')
    axes[1, 0].axis('off')

    # Training curves
    axes[1, 1].plot(history['test'], label='Test Loss', linewidth=2, color='blue')
    axes[1, 1].plot(history['consensus'], label='Consensus', linewidth=2, color='purple', alpha=0.7)
    axes[1, 1].plot(history['target'], label='Target', linewidth=2, color='orange', alpha=0.7)
    axes[1, 1].set_title('Training Curves', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('consensus_only_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'consensus_only_results.png'\n")
    plt.show()

    # Print stats
    print("="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"Test Loss:       {history['test'][-1]:.6f}")
    print(f"Consensus Loss:  {history['consensus'][-1]:.6f}")
    print(f"Target Loss:     {history['target'][-1]:.6f}")
    print("="*70)


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    # Example usage
    model, history = run_consensus_training(
        input_folder="/path/to/input/images",
        output_folder="/path/to/output/images",
        img_size=128,
        latent_dim=128,
        epochs=100,
        test_idx=6,
        structured_ratio=0.5,  # 50% structured, 50% random batches
        consensus_weight=0.6,
        target_weight=0.4
    )
