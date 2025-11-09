"""
CONSENSUS_BASIS_DIFFUSION - W/X/Y/Z Consensus with Iterative Refinement
========================================================================

Combines:
  - W/X/Y/Z tetrahedral basis structure
  - Consensus negotiation (X ↔ Y perspectives)
  - Diffusion-style iterative refinement
  - Pareto batching (20% structured)

Key Innovation:
  Instead of one-shot prediction, X and Y negotiate over T timesteps:
    Noise → [Step 1: X↔Y agree] → [Step 2: X↔Y agree] → ... → Consensus

  Annealed weighting:
    Early steps: 80% consensus, 20% target (build agreement)
    Late steps:  40% consensus, 60% target (refine accuracy)

Philosophy:
  "Reality emerges through PROCESS of negotiation, not instant decision."

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
import math
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom

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
# BASIS TRANSFORMS
# ============================================================================

def create_edge_version(img: torch.Tensor) -> torch.Tensor:
    """Edge representation (Y/Nonlinear basis)."""
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_3ch = np.stack([edges, edges, edges], axis=2) / 255.0
    return torch.from_numpy(edges_3ch).permute(2, 0, 1).float()

def create_grayscale_version(img: torch.Tensor) -> torch.Tensor:
    """Grayscale (X/Linear basis)."""
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_3ch = np.stack([gray, gray, gray], axis=2) / 255.0
    return torch.from_numpy(gray_3ch).permute(2, 0, 1).float()

def create_dithered_version(img: torch.Tensor) -> torch.Tensor:
    """Dithered (Y/Nonlinear discrete)."""
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    dithered = (gray > 128).astype(np.uint8) * 255
    dithered_3ch = np.stack([dithered, dithered, dithered], axis=2) / 255.0
    return torch.from_numpy(dithered_3ch).permute(2, 0, 1).float()


# ============================================================================
# BASIS DATASET
# ============================================================================

class BasisDataset(Dataset):
    """4 basis projections per sample: RGB, Edge, Grayscale, Dithered."""

    def __init__(self, image_pairs, apply_augmentation=True):
        self.base_pairs = []

        for input_img, output_img in image_pairs:
            if apply_augmentation:
                augs_in = self._augment(input_img)
                augs_out = self._augment(output_img)
                for aug_in, aug_out in zip(augs_in, augs_out):
                    self.base_pairs.append((aug_in, aug_out))
            else:
                self.base_pairs.append((input_img, output_img))

        self.n_base = len(self.base_pairs)
        print(f"  Base pairs: {self.n_base}, Total: {self.n_base * 4}")

    def _augment(self, img):
        """8 geometric transformations."""
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
        basis_type = idx % 4
        input_img, output_img = self.base_pairs[base_idx]

        if basis_type == 0:
            inp, out = input_img, output_img
        elif basis_type == 1:
            inp = create_edge_version(input_img)
            out = create_edge_version(output_img)
        elif basis_type == 2:
            inp = create_grayscale_version(input_img)
            out = create_grayscale_version(output_img)
        else:
            inp = create_dithered_version(input_img)
            out = create_dithered_version(output_img)

        return inp.reshape(-1), out.reshape(-1), base_idx, basis_type


# ============================================================================
# PARETO BATCH SAMPLER
# ============================================================================

class ParetoBatchSampler(Sampler):
    """Power-law structured batching (20% structured recommended)."""

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

        # Structured: same image, 4 basis views
        for i in range(n_structured):
            base_idx = base_indices[i]
            batch = [base_idx * 4 + b for b in range(4)]
            batches.append(batch)

        # Random: diverse samples
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
# DIFFUSION SCHEDULER
# ============================================================================

class ConsensusScheduler:
    """
    Anneals consensus/target weights during diffusion.

    Early timesteps (high noise):
        - High consensus weight (X and Y must agree)
        - Low target weight (don't worry about accuracy yet)

    Late timesteps (low noise):
        - Lower consensus weight (they already agree)
        - Higher target weight (refine toward accuracy)
    """

    def __init__(self, diffusion_steps=10,
                 initial_consensus=0.8, final_consensus=0.4,
                 initial_target=0.2, final_target=0.6):
        self.T = diffusion_steps
        self.c0 = initial_consensus
        self.c1 = final_consensus
        self.t0 = initial_target
        self.t1 = final_target

    def get_weights(self, t):
        """
        Get consensus and target weights for timestep t.

        Args:
            t: Current timestep (0 = noisy, T = clean)

        Returns:
            (consensus_weight, target_weight)
        """
        progress = t / self.T  # 0 → 1 as we denoise

        # Linearly interpolate
        consensus = self.c0 + (self.c1 - self.c0) * progress
        target = self.t0 + (self.t1 - self.t0) * progress

        return consensus, target

    def get_noise_scale(self, t):
        """Get noise level for timestep t."""
        # Cosine schedule (better than linear)
        s = 0.008  # offset
        f_t = math.cos(((t / self.T) + s) / (1 + s) * math.pi / 2) ** 2
        return f_t


# ============================================================================
# DIFFUSION CONSENSUS TRAINING
# ============================================================================

def train_diffusion_consensus(model, loader, optimizer, device,
                              latent_dim=128, scheduler=None):
    """
    Train with diffusion-based consensus.

    For each batch:
        1. Get target output
        2. Add noise at random timestep t
        3. X and Y denoise with consensus at timestep t
        4. Loss = weighted consensus + target
        5. Weights annealed based on t
    """
    if scheduler is None:
        scheduler = ConsensusScheduler(diffusion_steps=10)

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

        # Random diffusion timestep
        t = torch.randint(0, scheduler.T + 1, (bs,), device=device).float()

        # Add noise to target based on timestep
        noise = torch.randn_like(batch_y)
        noise_scales = torch.tensor([scheduler.get_noise_scale(ti.item())
                                    for ti in t], device=device).view(bs, 1)
        noisy_target = batch_y * noise_scales + noise * (1 - noise_scales)

        # Condition model on input + noisy target
        # (Model learns to denoise toward consensus)
        model_input = batch_x  # Could concatenate with noisy_target

        # === X and Y perspectives ===
        lin_v, lin_f = model.linear_net(model_input, return_faces=True)
        non_v, non_f = model.nonlinear_net(model_input, return_faces=True)

        # === Z-coupling ===
        coupled_l = [model.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i])
                     for i in range(4)]
        coupled_n = [model.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i])
                     for i in range(4)]

        lin_v = model.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
        non_v = model.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

        # === Separate outputs ===
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

        # === Annealed loss ===
        # Get weights based on average timestep in batch
        avg_t = t.mean().item()
        consensus_weight, target_weight = scheduler.get_weights(avg_t)

        # Consensus: X and Y must agree
        consensus_loss = F.mse_loss(linear_output, nonlinear_output)

        # Target: Denoise toward clean output
        target_loss = (F.mse_loss(linear_output, batch_y) +
                      F.mse_loss(nonlinear_output, batch_y)) / 2

        loss = consensus_weight * consensus_loss + target_weight * target_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_consensus += consensus_loss.item()
        total_target += target_loss.item()

    n = len(loader)
    return total_loss / n, total_consensus / n, total_target / n


def evaluate(model, loader, device):
    """Evaluate with direct prediction (no diffusion at test time yet)."""
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


# ============================================================================
# ITERATIVE INFERENCE (Diffusion Denoising)
# ============================================================================

def iterative_inference(model, input_img, device, latent_dim=128,
                       diffusion_steps=10, noise_scale=0.3):
    """
    Iterative refinement inference (like diffusion sampling).

    Args:
        model: Trained consensus model
        input_img: Input image tensor (flattened)
        diffusion_steps: Number of refinement iterations
        noise_scale: Initial noise level

    Returns:
        List of outputs at each step (for visualization)
    """
    model.eval()
    scheduler = ConsensusScheduler(diffusion_steps=diffusion_steps)

    # Start from noise + rough prediction
    with torch.no_grad():
        rough_output = model(input_img)

    # Add noise
    x_t = rough_output + torch.randn_like(rough_output) * noise_scale

    outputs = [x_t.clone()]

    with torch.no_grad():
        for t in range(diffusion_steps):
            # Get X and Y predictions
            lin_v, lin_f = model.linear_net(input_img, return_faces=True)
            non_v, non_f = model.nonlinear_net(input_img, return_faces=True)

            # Couple
            coupled_l = [model.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i])
                         for i in range(4)]
            coupled_n = [model.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i])
                         for i in range(4)]

            lin_v = model.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
            non_v = model.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

            # Get predictions
            bs = input_img.size(0)
            lin_only = torch.cat([lin_v.reshape(bs, -1),
                                 torch.zeros(bs, latent_dim * 4, device=device)], dim=-1)
            non_only = torch.cat([torch.zeros(bs, latent_dim * 4, device=device),
                                 non_v.reshape(bs, -1)], dim=-1)

            x_pred = model.output_projection(lin_only)
            y_pred = model.output_projection(non_only)

            # Consensus step: move toward agreement
            consensus_weight, _ = scheduler.get_weights(t)
            x_t = consensus_weight * ((x_pred + y_pred) / 2) + (1 - consensus_weight) * x_t

            outputs.append(x_t.clone())

    return outputs


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_diffusion_consensus(input_folder, output_folder,
                           img_size=128, latent_dim=128,
                           epochs=300, test_idx=6,
                           structured_ratio=0.2,
                           diffusion_steps=10,
                           device=None):
    """
    Run W/X/Y/Z diffusion consensus training.

    Returns:
        model, history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("🌀 W/X/Y/Z DIFFUSION CONSENSUS")
    print("="*70)
    print(f"Image size: {img_size}×{img_size}")
    print(f"Latent dim: {latent_dim}")
    print(f"Structured ratio: {structured_ratio} (Pareto)")
    print(f"Diffusion steps: {diffusion_steps}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    # Load data
    all_pairs = load_image_pairs(input_folder, output_folder, img_size)

    if len(all_pairs) == 0:
        raise ValueError("No image pairs found!")

    if test_idx >= len(all_pairs):
        test_idx = len(all_pairs) - 1

    train_pairs = [p for i, p in enumerate(all_pairs) if i != test_idx]
    test_pairs = [all_pairs[test_idx]]

    # Create datasets
    print("📦 Creating datasets...")
    train_dataset = BasisDataset(train_pairs, apply_augmentation=True)
    test_dataset = BasisDataset(test_pairs, apply_augmentation=True)

    # Create samplers
    train_sampler = ParetoBatchSampler(train_dataset, structured_ratio=structured_ratio)
    test_sampler = ParetoBatchSampler(test_dataset, structured_ratio=0.0, shuffle=False)

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
    scheduler = ConsensusScheduler(diffusion_steps=diffusion_steps)

    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Diffusion scheduler: {diffusion_steps} steps")
    print(f"   Consensus annealing: 0.8 → 0.4")
    print(f"   Target annealing: 0.2 → 0.6\n")

    # Train
    history = {'train': [], 'test': [], 'consensus': [], 'target': []}

    print("Training with diffusion consensus...")
    print("="*70)

    for epoch in range(epochs):
        train_loss, consensus, target = train_diffusion_consensus(
            model, train_loader, optimizer, device,
            latent_dim=latent_dim,
            scheduler=scheduler
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

    # Visualize with iterative refinement
    visualize_diffusion_results(model, train_dataset, history,
                                img_size, latent_dim, diffusion_steps, device)

    return model, history


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_diffusion_results(model, train_dataset, history,
                               img_size, latent_dim, diffusion_steps, device):
    """Visualize diffusion denoising process."""
    model.eval()

    # Get sample
    sample_x, sample_y = train_dataset.base_pairs[0]
    sample_x_tensor = sample_x.reshape(1, -1).to(device)

    # Get iterative outputs
    outputs = iterative_inference(model, sample_x_tensor, device,
                                  latent_dim, diffusion_steps)

    # Reshape to images
    def to_img(t):
        return t.reshape(3, img_size, img_size).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    input_img = to_img(sample_x)
    target_img = to_img(sample_y)

    # Plot diffusion steps
    n_show = min(6, len(outputs))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Top row
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_img)
    axes[0, 1].set_title('Target', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(to_img(outputs[-1]))
    axes[0, 2].set_title(f'Final Output\n(After {diffusion_steps} steps)',
                        fontsize=14, fontweight='bold', color='blue')
    axes[0, 2].axis('off')

    # Middle row: show denoising steps
    step_indices = [0, len(outputs)//3, 2*len(outputs)//3]
    for i, step_idx in enumerate(step_indices):
        axes[1, i].imshow(to_img(outputs[step_idx]))
        axes[1, i].set_title(f'Step {step_idx}/{diffusion_steps}',
                            fontsize=12, fontweight='bold')
        axes[1, i].axis('off')

    # Bottom row: training curves
    axes[2, 0].plot(history['test'], label='Test Loss', linewidth=2, color='blue')
    axes[2, 0].set_title('Test Loss', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(history['consensus'], label='Consensus', linewidth=2, color='purple')
    axes[2, 1].set_title('Consensus Loss (X↔Y)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].grid(alpha=0.3)

    axes[2, 2].plot(history['target'], label='Target', linewidth=2, color='orange')
    axes[2, 2].set_title('Target Loss', fontsize=12, fontweight='bold')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('diffusion_consensus_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'diffusion_consensus_results.png'\n")
    plt.show()

    # Stats
    print("="*70)
    print("📊 DIFFUSION CONSENSUS RESULTS")
    print("="*70)
    print(f"Diffusion steps: {diffusion_steps}")
    print(f"Test Loss:       {history['test'][-1]:.6f}")
    print(f"Consensus Loss:  {history['consensus'][-1]:.6f}")
    print(f"Target Loss:     {history['target'][-1]:.6f}")
    print("\n💡 Iterative refinement:")
    print(f"  • X and Y negotiate over {diffusion_steps} steps")
    print(f"  • Early: Build agreement (80% consensus)")
    print(f"  • Late: Refine accuracy (60% target)")
    print("="*70)


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    model, history = run_diffusion_consensus(
        input_folder="/path/to/input/images",
        output_folder="/path/to/output/images",
        img_size=128,
        latent_dim=128,
        epochs=300,
        test_idx=6,
        structured_ratio=0.2,  # Pareto-optimal
        diffusion_steps=10     # Iterative refinement steps
    )
