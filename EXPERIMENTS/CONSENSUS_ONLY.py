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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Assumes DualTetrahedralNetwork is already imported or available
# If not: from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Simple image pair dataset with flattened tensors."""

    def __init__(self, input_folder, output_folder, img_size=128):
        self.img_size = img_size
        self.pairs = self._load_pairs(input_folder, output_folder)

    def _load_pairs(self, input_folder, output_folder):
        """Load all image pairs from folders."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        input_files = sorted(input_path.glob('*.png')) + sorted(input_path.glob('*.jpg')) + sorted(input_path.glob('*.webp'))
        output_files = sorted(output_path.glob('*.png')) + sorted(output_path.glob('*.jpg')) + sorted(output_path.glob('*.webp'))

        pairs = []
        for inp, out in zip(input_files, output_files):
            img_in = Image.open(inp).convert('RGB').resize((self.img_size, self.img_size), Image.LANCZOS)
            img_out = Image.open(out).convert('RGB').resize((self.img_size, self.img_size), Image.LANCZOS)

            tensor_in = torch.from_numpy(np.array(img_in, dtype=np.float32) / 255.0).permute(2, 0, 1)
            tensor_out = torch.from_numpy(np.array(img_out, dtype=np.float32) / 255.0).permute(2, 0, 1)

            pairs.append((tensor_in, tensor_out))

        print(f"✓ Loaded {len(pairs)} image pairs")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, out = self.pairs[idx]
        return inp.reshape(-1), out.reshape(-1)


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

    for batch_x, batch_y in loader:
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
        for batch_x, batch_y in loader:
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
                          epochs=100, batch_size=4,
                          consensus_weight=0.6, target_weight=0.4,
                          test_split=0.15, device=None):
    """
    Run consensus training.

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
    print(f"Consensus weight: {consensus_weight}")
    print(f"Target weight: {target_weight}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("="*70)

    # Load data
    dataset = ImageDataset(input_folder, output_folder, img_size)

    # Split train/test
    n_test = max(1, int(len(dataset) * test_split))
    n_train = len(dataset) - n_test
    train_data, test_data = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Train: {n_train} | Test: {n_test}\n")

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
    visualize_results(model, dataset, history, img_size, device)

    return model, history


def visualize_results(model, dataset, history, img_size, device):
    """Visualize training results."""
    model.eval()

    # Get a sample
    sample_x, sample_y = dataset.pairs[0]
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
        consensus_weight=0.6,
        target_weight=0.4
    )
