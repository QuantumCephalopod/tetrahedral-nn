"""
CONSENSUS_MULTIREP_TRAINING - MSE vs Consensus Loss Comparison
==============================================================

Combines multi-representation dataset + hybrid batching + consensus loss
to test the philosophical hypothesis:

    "Reality emerges from negotiated agreement between perspectives,
     not from matching a privileged ground truth."

This experiment trains TWO models side-by-side:
  1. MSE Loss (traditional): "Match the target exactly"
  2. Consensus Loss (relational): "Linear and Nonlinear networks must agree"

Dataset:
  - Multi-representation: RGB, edges, grayscale, dithered (4 versions per sample)
  - Hybrid batching: Mix of structured (teaching) and random (experience)
  - 128×128 RGB images (fabric→skin or similar transformations)

Expected outcome:
  - Consensus should produce SHARPER outputs (less "mush")
  - Better network cooperation (Linear + Nonlinear agree more)
  - Possibly better generalization

Philosophy:
  MSE assumes the target IS reality. Consensus assumes reality emerges from
  perspective agreement. This tests process relationalism in neural loss design.

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
from typing import List, Tuple
import random
import cv2

# Import tetrahedral architecture
import sys
from pathlib import Path

# Add repo root to path (for both Colab and local use)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork


# ============================================================================
# IMAGE LOADING
# ============================================================================

class ImagePairLoader:
    """
    Load image pairs from folders based on sorted filenames.

    Handles: webp, png, jpg, jpeg
    Output: Normalized RGB tensors
    """
    def __init__(self, input_folder: str, output_folder: str, target_size: int = 128):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.target_size = target_size

        # Get sorted list of image files
        self.input_files = sorted(self._get_image_files(self.input_folder))
        self.output_files = sorted(self._get_image_files(self.output_folder))

        if len(self.input_files) != len(self.output_files):
            raise ValueError(
                f"Mismatched number of files: {len(self.input_files)} in input, "
                f"{len(self.output_files)} in output folder."
            )

        if len(self.input_files) == 0:
            raise ValueError(f"No image files found in {input_folder} or {output_folder}")

    def _get_image_files(self, folder: Path) -> List[Path]:
        """Get list of all image files in a folder."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        files = []
        for ext in extensions:
            files.extend(list(folder.glob(ext)))
        return files

    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load image and convert to tensor.

        Returns:
            Tensor of shape (3, target_size, target_size) normalized to [0, 1]
        """
        img = Image.open(path).convert('RGB')
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor

    def load_all_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load all image pairs."""
        pairs = []
        num_pairs = len(self.input_files)
        print(f"📂 Loading {num_pairs} image pairs...")

        for i in range(num_pairs):
            pair = (self._load_image(self.input_files[i]),
                   self._load_image(self.output_files[i]))
            pairs.append(pair)
            print(f"  ✓ Pair {i+1}/{num_pairs}")

        return pairs


# ============================================================================
# MULTI-REPRESENTATION FUNCTIONS
# ============================================================================

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


# ============================================================================
# MULTI-REPRESENTATION DATASET
# ============================================================================

class MultiRepresentationDataset(Dataset):
    """
    Dataset with 4 representations per sample:
      0. Original RGB
      1. Edge-detected
      2. Grayscale
      3. Dithered

    This forces the network to learn transformation STRUCTURE, not pixel patterns.
    """
    def __init__(self, image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                 apply_augmentation: bool = True):
        self.base_pairs = []

        # Simple augmentation (4 rotations × 2 flips)
        for input_img, output_img in image_pairs:
            if apply_augmentation:
                augs_in = self.augment_image_simple(input_img)
                augs_out = self.augment_image_simple(output_img)
                for aug_in, aug_out in zip(augs_in, augs_out):
                    self.base_pairs.append((aug_in, aug_out))
            else:
                self.base_pairs.append((input_img, output_img))

        self.n_base = len(self.base_pairs)
        print(f"  Base pairs: {self.n_base}, Total samples: {self.n_base * 4}")

    def augment_image_simple(self, img: torch.Tensor) -> List[torch.Tensor]:
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

        # Create representation based on type
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


# ============================================================================
# HYBRID BATCH SAMPLER
# ============================================================================

class HybridBatchSampler(Sampler):
    """
    Mix structured and random batches.

    structured_ratio=1.0: All batches have same base pair, 4 reps (teaching)
    structured_ratio=0.5: 50% structured, 50% random (balanced)
    structured_ratio=0.0: Fully random batches (real-world experience)
    """
    def __init__(self, dataset, structured_ratio: float = 0.5, shuffle: bool = True):
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, optimizer, device, use_consensus=False,
                   consensus_weight=0.6, target_weight=0.4,
                   input_recon_weight=0.1, latent_dim=128):
    """
    Train for one epoch with either MSE or consensus loss.

    Args:
        use_consensus: If True, use consensus loss. If False, use MSE.
        consensus_weight: Weight for inter-network agreement (default 0.6)
        target_weight: Weight for target guidance (default 0.4)
        input_recon_weight: Weight for input reconstruction (default 0.1)
    """
    model.train()
    epoch_loss = 0.0
    epoch_consensus = 0.0
    epoch_target = 0.0
    n_batches = 0

    for batch_data in loader:
        batch_x, batch_y, _, _ = batch_data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        bs = batch_x.size(0)

        optimizer.zero_grad()

        # Forward pass through both networks
        lin_v, lin_f = model.linear_net(batch_x, return_faces=True)
        non_v, non_f = model.nonlinear_net(batch_x, return_faces=True)

        # Face coupling
        coupled_l = [model.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i]) for i in range(4)]
        coupled_n = [model.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i]) for i in range(4)]

        lin_v = model.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
        non_v = model.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

        vertices = torch.cat([lin_v.reshape(bs, -1), non_v.reshape(bs, -1)], dim=-1)

        # Get outputs
        output = model.output_projection(vertices)

        # Input reconstruction
        if hasattr(model, 'input_reconstructor'):
            recon = model.input_reconstructor(vertices)
            recon_loss = F.mse_loss(recon, batch_x)
        else:
            recon_loss = 0.0

        if use_consensus:
            # CONSENSUS LOSS: Networks must agree
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

            # Consensus: how much do perspectives agree?
            consensus_loss = F.mse_loss(linear_output, nonlinear_output)

            # Target guidance: both networks orbit around target
            target_loss = (F.mse_loss(linear_output, batch_y) +
                          F.mse_loss(nonlinear_output, batch_y)) / 2

            # Combined
            main_loss = consensus_weight * consensus_loss + target_weight * target_loss

            epoch_consensus += consensus_loss.item()
            epoch_target += target_loss.item()
        else:
            # STANDARD MSE LOSS
            main_loss = F.mse_loss(output, batch_y)

        # Total loss
        total_loss = main_loss + input_recon_weight * recon_loss

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        n_batches += 1

    return {
        'loss': epoch_loss / n_batches,
        'consensus': epoch_consensus / n_batches if use_consensus else 0.0,
        'target': epoch_target / n_batches if use_consensus else 0.0
    }


def evaluate(model, loader, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_x, batch_y, _, _ = batch_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            test_loss += F.mse_loss(output, batch_y).item()
            n_batches += 1

    return test_loss / n_batches


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(input_folder: str, output_folder: str,
                  img_size: int = 128, latent_dim: int = 128,
                  epochs: int = 100, test_idx: int = 6,
                  structured_ratio: float = 0.5,
                  consensus_weight: float = 0.6,
                  target_weight: float = 0.4,
                  device: str = None):
    """
    Run MSE vs Consensus comparison experiment.

    Args:
        input_folder: Path to input images
        output_folder: Path to output images
        img_size: Image size (square)
        latent_dim: Latent dimension for tetrahedra
        epochs: Number of training epochs
        test_idx: Which pair to hold out for testing
        structured_ratio: 0.0-1.0, ratio of structured batches
        consensus_weight: Weight for consensus loss term
        target_weight: Weight for target guidance term
        device: 'cuda' or 'cpu' (auto-detected if None)

    Returns:
        (model_mse, model_consensus, history_mse, history_consensus)
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("🌀 CONSENSUS vs MSE: Multi-Rep + Hybrid Batching")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Image size: {img_size}×{img_size}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Structured ratio: {structured_ratio}")
    print(f"  Consensus weight: {consensus_weight}")
    print(f"  Target weight: {target_weight}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")
    print(f"\n{'='*70}\n")

    # Load data
    loader = ImagePairLoader(input_folder, output_folder, target_size=img_size)
    all_pairs = loader.load_all_pairs()

    train_pairs = [p for i, p in enumerate(all_pairs) if i != test_idx]
    test_pairs = [all_pairs[test_idx]]

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = MultiRepresentationDataset(train_pairs, apply_augmentation=True)
    test_dataset = MultiRepresentationDataset(test_pairs, apply_augmentation=True)

    # Create samplers
    train_sampler = HybridBatchSampler(train_dataset, structured_ratio=structured_ratio)
    test_sampler = HybridBatchSampler(test_dataset, structured_ratio=0.0, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}\n")

    # Create models
    print("🏗️  Building models...")
    input_dim = img_size * img_size * 3

    model_mse = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=latent_dim,
        coupling_strength=0.5,
        output_mode="weighted"
    ).to(device)
    model_mse.input_reconstructor = nn.Linear(latent_dim * 8, input_dim).to(device)

    model_consensus = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=latent_dim,
        coupling_strength=0.5,
        output_mode="weighted"
    ).to(device)
    model_consensus.input_reconstructor = nn.Linear(latent_dim * 8, input_dim).to(device)

    print(f"✓ Models ready ({sum(p.numel() for p in model_mse.parameters()):,} params)\n")

    # Train MSE model
    print("⚡ Training MSE model...")
    print("="*70)
    optimizer_mse = optim.Adam(model_mse.parameters(), lr=0.0001)
    history_mse = {'train': [], 'test': []}

    for epoch in range(epochs):
        metrics = train_one_epoch(model_mse, train_loader, optimizer_mse, device,
                                 use_consensus=False, latent_dim=latent_dim)
        test_loss = evaluate(model_mse, test_loader, device)

        history_mse['train'].append(metrics['loss'])
        history_mse['test'].append(test_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {metrics['loss']:.6f} | Test: {test_loss:.6f}")

    # Train Consensus model
    print("\n⚡ Training Consensus model...")
    print("="*70)
    optimizer_consensus = optim.Adam(model_consensus.parameters(), lr=0.0001)
    history_consensus = {'train': [], 'test': [], 'consensus': [], 'target': []}

    for epoch in range(epochs):
        metrics = train_one_epoch(model_consensus, train_loader, optimizer_consensus, device,
                                 use_consensus=True, consensus_weight=consensus_weight,
                                 target_weight=target_weight, latent_dim=latent_dim)
        test_loss = evaluate(model_consensus, test_loader, device)

        history_consensus['train'].append(metrics['loss'])
        history_consensus['test'].append(test_loss)
        history_consensus['consensus'].append(metrics['consensus'])
        history_consensus['target'].append(metrics['target'])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {metrics['loss']:.6f} | "
                  f"Test: {test_loss:.6f} | Consensus: {metrics['consensus']:.6f}")

    # Visualize results
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATION")
    print("="*70)

    visualize_comparison(
        model_mse, model_consensus,
        train_dataset, test_dataset,
        history_mse, history_consensus,
        img_size, latent_dim, device
    )

    # Print results
    print_results(history_mse, history_consensus)

    return model_mse, model_consensus, history_mse, history_consensus


def visualize_comparison(model_mse, model_consensus, train_dataset, test_dataset,
                        history_mse, history_consensus, img_size, latent_dim, device):
    """Generate comparison visualization."""

    # Get test sample
    test_input = train_dataset.base_pairs[0][0]
    test_target = train_dataset.base_pairs[0][1]
    test_input_tensor = test_input.unsqueeze(0).to(device)
    test_target_tensor = test_target.unsqueeze(0).to(device)

    model_mse.eval()
    model_consensus.eval()

    with torch.no_grad():
        # MSE output
        mse_output = model_mse(test_input_tensor).cpu()

        # Consensus outputs
        bs = 1
        lin_v, lin_f = model_consensus.linear_net(test_input_tensor, return_faces=True)
        non_v, non_f = model_consensus.nonlinear_net(test_input_tensor, return_faces=True)

        coupled_l = [model_consensus.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i]) for i in range(4)]
        coupled_n = [model_consensus.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i]) for i in range(4)]

        lin_v = model_consensus.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
        non_v = model_consensus.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

        # Separate outputs
        lin_vertices_only = torch.cat([
            lin_v.reshape(bs, -1),
            torch.zeros(bs, latent_dim * 4, device=device)
        ], dim=-1)
        non_vertices_only = torch.cat([
            torch.zeros(bs, latent_dim * 4, device=device),
            non_v.reshape(bs, -1)
        ], dim=-1)

        cons_linear = model_consensus.output_projection(lin_vertices_only).cpu()
        cons_nonlinear = model_consensus.output_projection(non_vertices_only).cpu()
        cons_output = model_consensus(test_input_tensor).cpu()

    # Reshape to images
    def to_img(tensor):
        img = tensor.reshape(3, img_size, img_size).permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)

    input_img = to_img(test_input)
    target_img = to_img(test_target)
    mse_img = to_img(mse_output)
    cons_linear_img = to_img(cons_linear)
    cons_nonlinear_img = to_img(cons_nonlinear)
    cons_img = to_img(cons_output)

    # Calculate metrics
    mse_loss_val = F.mse_loss(mse_output, test_target_tensor.cpu()).item()
    cons_loss_val = F.mse_loss(cons_output, test_target_tensor.cpu()).item()
    agreement = F.mse_loss(cons_linear, cons_nonlinear).item()
    mse_sharpness = mse_img.var()
    cons_sharpness = cons_img.var()

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_img)
    axes[0, 1].set_title('Target', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mse_img)
    axes[0, 2].set_title(f'MSE Output\nLoss: {mse_loss_val:.4f} | Var: {mse_sharpness:.4f}',
                         fontsize=11, fontweight='bold', color='red')
    axes[0, 2].axis('off')

    axes[0, 3].plot(history_mse['test'], label='MSE', color='red', linewidth=2, alpha=0.7)
    axes[0, 3].plot(history_consensus['test'], label='Consensus', color='blue', linewidth=2, alpha=0.7)
    axes[0, 3].set_title('Test Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('Loss')
    axes[0, 3].legend()
    axes[0, 3].grid(alpha=0.3)

    # Bottom row
    axes[1, 0].imshow(cons_linear_img)
    axes[1, 0].set_title('Consensus: Linear\n(Smooth)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cons_nonlinear_img)
    axes[1, 1].set_title('Consensus: Nonlinear\n(Boundary)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cons_img)
    axes[1, 2].set_title(f'Consensus Output\nLoss: {cons_loss_val:.4f} | Var: {cons_sharpness:.4f}\nAgreement: {agreement:.4f}',
                         fontsize=10, fontweight='bold', color='blue')
    axes[1, 2].axis('off')

    axes[1, 3].plot(history_consensus['consensus'], label='Consensus',
                    color='purple', linewidth=2, alpha=0.7)
    axes[1, 3].plot(history_consensus['target'], label='Target',
                    color='orange', linewidth=2, alpha=0.7)
    axes[1, 3].set_title('Consensus Breakdown', fontsize=12, fontweight='bold')
    axes[1, 3].set_xlabel('Epoch')
    axes[1, 3].set_ylabel('Loss')
    axes[1, 3].legend()
    axes[1, 3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('consensus_multirep_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'consensus_multirep_comparison.png'\n")
    plt.show()


def print_results(history_mse, history_consensus):
    """Print experiment results."""
    print("="*70)
    print("📊 FINAL RESULTS")
    print("="*70)

    print(f"\nTest Loss (Lower = Better):")
    print(f"  MSE:       {history_mse['test'][-1]:.6f}")
    print(f"  Consensus: {history_consensus['test'][-1]:.6f}")

    improvement = ((history_mse['test'][-1] - history_consensus['test'][-1]) / history_mse['test'][-1]) * 100
    if history_consensus['test'][-1] < history_mse['test'][-1]:
        print(f"  → Consensus is {improvement:.1f}% better! ✓")
    else:
        print(f"  → MSE is {-improvement:.1f}% better")

    print("\n💡 INTERPRETATION:")
    if history_consensus['test'][-1] < history_mse['test'][-1]:
        print("  ✓ CONSENSUS WINS on test loss")
        print("  ✓ Reality as negotiation > reality as ground truth")
        print("  ✓ The two perspectives learned to cooperate")
    else:
        print("  ~ MSE has lower test loss")
        print("  → Look at visual quality - consensus may be sharper")
        print("  → Try adjusting consensus_weight hyperparameter")

    print("\n🎨 VISUAL INSPECTION:")
    print("  Check the generated image - which looks better?")
    print("  - Does Consensus have more defined features?")
    print("  - Does MSE look averaged/mushy?")
    print("  - Do Linear/Nonlinear show complementary perspectives?")

    print("\n" + "="*70)


# ============================================================================
# COLAB QUICKSTART
# ============================================================================

def colab_quickstart(epochs: int = 100):
    """
    Quick start for Google Colab.

    Update the paths below to your image folders, then run:
        model_mse, model_consensus, history_mse, history_consensus = colab_quickstart()
    """

    # UPDATE THESE PATHS
    input_folder = "/content/drive/MyDrive/trainingdata/images_49/set_04"
    output_folder = "/content/drive/MyDrive/trainingdata/images_49/set_05"

    return run_experiment(
        input_folder=input_folder,
        output_folder=output_folder,
        img_size=128,
        latent_dim=128,
        epochs=epochs,
        test_idx=6,
        structured_ratio=0.5,
        consensus_weight=0.6,
        target_weight=0.4
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONSENSUS_MULTIREP_TRAINING.py")
    print("="*70)
    print("\n📖 This experiment tests:")
    print("  MSE Loss: 'Reality is the target, match it exactly'")
    print("  Consensus Loss: 'Reality emerges from perspective agreement'")
    print("\n🚀 For Colab:")
    print("  1. Update paths in colab_quickstart()")
    print("  2. Run: results = colab_quickstart()")
    print("\n💡 Philosophy:")
    print("  Process relationalism in neural loss design")
    print("  Testing if negotiated truth > privileged truth")
    print("="*70 + "\n")
