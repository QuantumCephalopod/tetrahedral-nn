"""
CONSENSUS_BASIS_TRANSFORM - W/X/Y/Z Consensus Image Transformation
===================================================================

Single-cell experiment combining:
  - W (Geometry): Tetrahedral basis structure
  - X (Linear): Smooth manifold perspective
  - Y (Nonlinear): Boundary detection perspective
  - Z (Coupling): Inter-face consensus negotiation
  - Pareto batching: Low structured ratio (20%) for natural learning

Philosophy:
  "Reality emerges from negotiated agreement between complementary
   geometric perspectives, not from matching privileged ground truth."

This is the CLEAN version - no MSE comparison, pure consensus.

Author: Philipp Remy Bartholomäus
Date: November 2025
"""

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

# Assumes DualTetrahedralNetwork is already imported
# If not: from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image_pairs(input_folder, output_folder, img_size=128):
    """Load all image pairs from folders."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    input_files = sorted(list(input_path.glob('*.png')) +
                        list(input_path.glob('*.jpg')) +
                        list(input_path.glob('*.webp')))
    output_files = sorted(list(output_path.glob('*.png')) +
                         list(output_path.glob('*.jpg')) +
                         list(output_path.glob('*.webp')))

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


# ============================================================================
# MULTI-REPRESENTATION TRANSFORMS (Basis Projections)
# ============================================================================

def create_edge_version(img: torch.Tensor) -> torch.Tensor:
    """Edge representation - emphasizes boundaries (Y/Nonlinear basis)."""
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_3ch = np.stack([edges, edges, edges], axis=2) / 255.0
    return torch.from_numpy(edges_3ch).permute(2, 0, 1).float()


def create_grayscale_version(img: torch.Tensor) -> torch.Tensor:
    """Grayscale representation - smooth intensity (X/Linear basis)."""
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_3ch = np.stack([gray, gray, gray], axis=2) / 255.0
    return torch.from_numpy(gray_3ch).permute(2, 0, 1).float()


def create_dithered_version(img: torch.Tensor) -> torch.Tensor:
    """Binary dithered - discrete decision boundaries (Y/Nonlinear basis)."""
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    dithered = (gray > 128).astype(np.uint8) * 255
    dithered_3ch = np.stack([dithered, dithered, dithered], axis=2) / 255.0
    return torch.from_numpy(dithered_3ch).permute(2, 0, 1).float()


# ============================================================================
# BASIS-AWARE DATASET
# ============================================================================

class BasisDataset(Dataset):
    """
    Dataset with 4 basis projections per sample:
      0. RGB (Full W basis)
      1. Edge (Y/Nonlinear emphasis)
      2. Grayscale (X/Linear emphasis)
      3. Dithered (Y/Nonlinear discrete)

    This teaches the network to see through different geometric lenses.
    """

    def __init__(self, image_pairs, apply_augmentation=True):
        self.base_pairs = []

        # Geometric augmentation (4 rotations × 2 flips = 8)
        for input_img, output_img in image_pairs:
            if apply_augmentation:
                augs_in = self._augment(input_img)
                augs_out = self._augment(output_img)
                for aug_in, aug_out in zip(augs_in, augs_out):
                    self.base_pairs.append((aug_in, aug_out))
            else:
                self.base_pairs.append((input_img, output_img))

        self.n_base = len(self.base_pairs)
        print(f"  Base pairs: {self.n_base}")
        print(f"  Total samples: {self.n_base * 4} (×4 basis projections)")

    def _augment(self, img):
        """8 geometric transformations preserving tetrahedral symmetry."""
        augs = []
        for k in range(4):  # 4 rotations
            rotated = torch.rot90(img, k=k, dims=[1, 2])
            augs.append(rotated)
            augs.append(torch.flip(rotated, dims=[2]))  # + mirror
        return augs

    def __len__(self):
        return self.n_base * 4

    def __getitem__(self, idx):
        base_idx = idx // 4
        basis_type = idx % 4  # Which basis projection
        input_img, output_img = self.base_pairs[base_idx]

        # Project onto basis
        if basis_type == 0:
            inp, out = input_img, output_img  # RGB (full)
        elif basis_type == 1:
            inp = create_edge_version(input_img)  # Y basis
            out = create_edge_version(output_img)
        elif basis_type == 2:
            inp = create_grayscale_version(input_img)  # X basis
            out = create_grayscale_version(output_img)
        else:
            inp = create_dithered_version(input_img)  # Y basis (discrete)
            out = create_dithered_version(output_img)

        return inp.reshape(-1), out.reshape(-1), base_idx, basis_type


# ============================================================================
# PARETO BATCH SAMPLER (Power-law structured batching)
# ============================================================================

class ParetoBatchSampler(Sampler):
    """
    Pareto-distributed structured batching.

    Low structured_ratio (e.g., 0.2) creates power-law distribution:
      - Few highly structured batches (same image, 4 basis views)
      - Many random batches (diverse experience)

    This mimics natural learning: occasional deep focus, mostly diverse exploration.

    Args:
        structured_ratio: 0.0-1.0
            0.2 = 20% structured (Pareto-optimal)
            0.5 = balanced
            1.0 = all structured
    """

    def __init__(self, dataset, structured_ratio=0.2, shuffle=True):
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

        # Structured batches: same image, 4 basis projections
        # This is "teaching mode" - network sees complementary views
        for i in range(n_structured):
            base_idx = base_indices[i]
            batch = [base_idx * 4 + b for b in range(4)]
            batches.append(batch)

        # Random batches: diverse samples
        # This is "experience mode" - network generalizes
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


# ============================================================================
# CONSENSUS TRAINING (X ↔ Y Negotiation)
# ============================================================================

def train_consensus_step(model, loader, optimizer, device, latent_dim=128,
                         consensus_weight=0.65, target_weight=0.35):
    """
    One epoch of consensus training.

    The two networks (X/Linear and Y/Nonlinear) negotiate reality:
      - 65%: Internal coherence (they must agree)
      - 35%: External coherence (orbit around target)

    This is Z-coupling in action: faces negotiate through attention.
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

        # === FORWARD: Both perspectives process input ===
        lin_v, lin_f = model.linear_net(batch_x, return_faces=True)
        non_v, non_f = model.nonlinear_net(batch_x, return_faces=True)

        # === Z-COUPLING: Face-to-face negotiation ===
        coupled_l = [model.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i]) for i in range(4)]
        coupled_n = [model.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i]) for i in range(4)]

        # Update vertices from face coupling
        lin_v = model.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
        non_v = model.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

        # === GET SEPARATE OUTPUTS (key to consensus!) ===
        # Linear perspective sees only its own vertices
        lin_vertices_only = torch.cat([
            lin_v.reshape(bs, -1),
            torch.zeros(bs, latent_dim * 4, device=device)
        ], dim=-1)

        # Nonlinear perspective sees only its own vertices
        non_vertices_only = torch.cat([
            torch.zeros(bs, latent_dim * 4, device=device),
            non_v.reshape(bs, -1)
        ], dim=-1)

        linear_output = model.output_projection(lin_vertices_only)
        nonlinear_output = model.output_projection(non_vertices_only)

        # === CONSENSUS LOSS: Perspectives must agree (internal coherence) ===
        consensus_loss = F.mse_loss(linear_output, nonlinear_output)

        # === TARGET LOSS: Both orbit around ground truth (external coherence) ===
        target_loss = (F.mse_loss(linear_output, batch_y) +
                      F.mse_loss(nonlinear_output, batch_y)) / 2

        # === WEIGHTED COMBINATION ===
        # More weight on agreement than on matching target!
        loss = consensus_weight * consensus_loss + target_weight * target_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_consensus += consensus_loss.item()
        total_target += target_loss.item()

    n = len(loader)
    return total_loss / n, total_consensus / n, total_target / n


def evaluate(model, loader, device):
    """Evaluate using combined output (not separate perspectives)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_data in loader:
            batch_x, batch_y, _, _ = batch_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Combined output (what the user sees)
            output = model(batch_x)
            total_loss += F.mse_loss(output, batch_y).item()

    return total_loss / len(loader)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_basis_consensus(input_folder, output_folder,
                       img_size=128, latent_dim=128,
                       epochs=300, test_idx=6,
                       structured_ratio=0.2,  # Pareto-optimal
                       consensus_weight=0.65, target_weight=0.35,
                       device=None):
    """
    Run W/X/Y/Z basis consensus training.

    Args:
        structured_ratio: 0.2 recommended (Pareto principle)
        consensus_weight: 0.65 = agreement more important than target
        target_weight: 0.35 = external guidance

    Returns:
        model, history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("🌀 W/X/Y/Z BASIS CONSENSUS TRAINING")
    print("="*70)
    print(f"Image size: {img_size}×{img_size}")
    print(f"Latent dim: {latent_dim}")
    print(f"Structured ratio: {structured_ratio} (Pareto)")
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

    # Create basis-aware datasets
    print("📦 Creating basis-aware datasets...")
    train_dataset = BasisDataset(train_pairs, apply_augmentation=True)
    test_dataset = BasisDataset(test_pairs, apply_augmentation=True)

    # Create Pareto batch samplers
    train_sampler = ParetoBatchSampler(train_dataset, structured_ratio=structured_ratio)
    test_sampler = ParetoBatchSampler(test_dataset, structured_ratio=0.0, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}\n")

    # Create W/X/Y/Z model
    input_dim = img_size * img_size * 3
    model = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=latent_dim,
        coupling_strength=0.5,
        output_mode="weighted"
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   W: Geometric basis (4 vertices, 6 edges, 4 faces)")
    print(f"   X: Linear tetrahedron (smooth manifolds)")
    print(f"   Y: Nonlinear tetrahedron (boundaries)")
    print(f"   Z: Face coupling (consensus negotiation)\n")

    # Train
    history = {'train': [], 'test': [], 'consensus': [], 'target': []}

    print("Training...")
    print("="*70)

    for epoch in range(epochs):
        train_loss, consensus, target = train_consensus_step(
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
    visualize_basis_results(model, train_dataset, history, img_size, latent_dim, device)

    return model, history


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_basis_results(model, train_dataset, history, img_size, latent_dim, device):
    """Visualize results showing X/Y perspectives."""
    model.eval()

    # Get sample
    sample_x, sample_y = train_dataset.base_pairs[0]
    sample_x_tensor = sample_x.reshape(1, -1).to(device)

    with torch.no_grad():
        # Get separate X and Y perspectives
        lin_v, _ = model.linear_net(sample_x_tensor, return_faces=True)
        non_v, _ = model.nonlinear_net(sample_x_tensor, return_faces=True)

        # X perspective output
        lin_only = torch.cat([
            lin_v.reshape(1, -1),
            torch.zeros(1, latent_dim * 4, device=device)
        ], dim=-1)
        x_output = model.output_projection(lin_only).cpu()

        # Y perspective output
        non_only = torch.cat([
            torch.zeros(1, latent_dim * 4, device=device),
            non_v.reshape(1, -1)
        ], dim=-1)
        y_output = model.output_projection(non_only).cpu()

        # Combined consensus output
        consensus_output = model(sample_x_tensor).cpu()

    # Reshape to images
    def to_img(t):
        return t.reshape(3, img_size, img_size).permute(1, 2, 0).numpy().clip(0, 1)

    input_img = to_img(sample_x)
    target_img = to_img(sample_y)
    x_img = to_img(x_output)
    y_img = to_img(y_output)
    consensus_img = to_img(consensus_output)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_img)
    axes[0, 1].set_title('Target', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(consensus_img)
    axes[0, 2].set_title('Consensus Output\n(Z-Coupled)',
                        fontsize=14, fontweight='bold', color='blue')
    axes[0, 2].axis('off')

    # Bottom row
    axes[1, 0].imshow(x_img)
    axes[1, 0].set_title('X Perspective\n(Linear/Smooth)',
                        fontsize=13, fontweight='bold', color='green')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(y_img)
    axes[1, 1].set_title('Y Perspective\n(Nonlinear/Boundary)',
                        fontsize=13, fontweight='bold', color='red')
    axes[1, 1].axis('off')

    # Training curves
    axes[1, 2].plot(history['test'], label='Test Loss', linewidth=2, color='blue')
    axes[1, 2].plot(history['consensus'], label='Consensus (X↔Y)',
                   linewidth=2, color='purple', alpha=0.7)
    axes[1, 2].plot(history['target'], label='Target Guidance',
                   linewidth=2, color='orange', alpha=0.7)
    axes[1, 2].set_title('Training Dynamics', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('consensus_basis_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'consensus_basis_results.png'\n")
    plt.show()

    # Print stats
    print("="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    print(f"Test Loss:       {history['test'][-1]:.6f}")
    print(f"Consensus Loss:  {history['consensus'][-1]:.6f}")
    print(f"Target Loss:     {history['target'][-1]:.6f}")
    print(f"\nConsensus reduction: {history['consensus'][0]:.6f} → {history['consensus'][-1]:.6f}")
    print(f"Target reduction:    {history['target'][0]:.6f} → {history['target'][-1]:.6f}")
    print("="*70)
    print("\n💡 Interpretation:")
    print("  • Consensus ↓ = X and Y perspectives converging")
    print("  • Target ↓ = Both perspectives approaching ground truth")
    print("  • Low consensus = Geometric coherence achieved!")
    print("="*70)


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    # Example usage
    model, history = run_basis_consensus(
        input_folder="/path/to/input/images",
        output_folder="/path/to/output/images",
        img_size=128,
        latent_dim=128,
        epochs=300,
        test_idx=6,
        structured_ratio=0.2,  # Pareto-optimal (20% structured, 80% random)
        consensus_weight=0.65,  # Agreement > matching
        target_weight=0.35
    )
