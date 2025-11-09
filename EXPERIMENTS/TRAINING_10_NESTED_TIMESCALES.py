"""
TRAINING_10 - NESTED TIMESCALES + BLENDED MSE→SSIM
===================================================

Unified training combining all discoveries:
  1. Blended loss: MSE→SSIM (100% MSE → 20% MSE / 80% SSIM)
  2. Nested timescales: Morphogenetic hierarchy via learning rates
  3. Consensus + diffusion annealing
  4. Pareto batching + basis representations
  5. Checkpoint saving for continual learning

Philosophy:
  "What is worth predicting?"

  - MSE bootstraps structure (can't learn from pure noise with SSIM)
  - SSIM refines perceptual quality once structure exists
  - Nested timescales: Field (coupling) stable, activity (vertices) reactive
  - Self-organization: No pre-assignment of roles
  - Active inference: The loss landscape defines what "matters"

Author: Philipp Remy Bartholomäus
Date: November 9, 2025
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
from typing import Optional

# Assumes core architecture available (W, X, Y, Z from IMAGE_TRANSFORM cell)
# from W_FOUNDATION.W_geometry import *
# from X_LINEAR.X_linear_tetrahedron import LinearTetrahedron
# from Y_NONLINEAR.Y_nonlinear_tetrahedron import NonlinearTetrahedron
# from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork


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
        """Get consensus and target weights for timestep t."""
        progress = t / self.T
        consensus = self.c0 + (self.c1 - self.c0) * progress
        target = self.t0 + (self.t1 - self.t0) * progress
        return consensus, target

    def get_noise_scale(self, t):
        """Get noise level for timestep t (cosine schedule)."""
        s = 0.008
        f_t = math.cos(((t / self.T) + s) / (1 + s) * math.pi / 2) ** 2
        return f_t


# ============================================================================
# BLENDED LOSS SCHEDULER
# ============================================================================

class BlendedLossScheduler:
    """
    Blends MSE→SSIM during training.

    Mirrors Pareto ratio:
      - Start: 100% MSE, 0% SSIM (bootstrap structure)
      - End:   20% MSE, 80% SSIM (refine perception)
    """

    def __init__(self, bootstrap_epochs=30, total_epochs=300):
        self.bootstrap_epochs = bootstrap_epochs
        self.total_epochs = total_epochs

    def get_weights(self, epoch):
        """
        Get MSE and SSIM weights for current epoch.

        Returns:
            (mse_weight, ssim_weight)
        """
        if epoch < self.bootstrap_epochs:
            # Pure MSE during bootstrap
            return 1.0, 0.0

        # Blend from (1.0, 0.0) → (0.2, 0.8)
        progress = (epoch - self.bootstrap_epochs) / (self.total_epochs - self.bootstrap_epochs)
        progress = min(progress, 1.0)  # Cap at 1.0

        mse_weight = 1.0 - (0.8 * progress)   # 1.0 → 0.2
        ssim_weight = 0.8 * progress           # 0.0 → 0.8

        return mse_weight, ssim_weight


# ============================================================================
# SSIM LOSS
# ============================================================================

def ssim_loss(pred, target, img_size, window_size=11):
    """
    Differentiable SSIM-based loss function.

    Returns (1 - SSIM) to use as loss (minimize).
    """
    bs = pred.size(0)
    pred_img = pred.reshape(bs, 3, img_size, img_size)
    target_img = target.reshape(bs, 3, img_size, img_size)

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    window = gaussian_window(window_size).to(pred_img.device)
    window = window.unsqueeze(0).unsqueeze(0)

    def apply_filter(img, window):
        C = img.size(1)
        _window = window.expand(C, 1, -1, -1)
        return F.conv2d(img, _window, padding=window_size // 2, groups=C)

    mu1 = apply_filter(pred_img, window)
    mu2 = apply_filter(target_img, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = apply_filter(pred_img ** 2, window) - mu1_sq
    sigma2_sq = apply_filter(target_img ** 2, window) - mu2_sq
    sigma12 = apply_filter(pred_img * target_img, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_val = ssim_map.mean()
    return 1.0 - ssim_val


# ============================================================================
# NESTED TIMESCALE TRAINING
# ============================================================================

def train_nested_consensus(model, loader, optimizer, device, epoch,
                          latent_dim=128, img_size=128,
                          consensus_scheduler=None,
                          loss_scheduler=None):
    """
    Train with nested timescales + blended MSE→SSIM consensus.
    """
    if consensus_scheduler is None:
        consensus_scheduler = ConsensusScheduler(diffusion_steps=10)
    if loss_scheduler is None:
        loss_scheduler = BlendedLossScheduler()

    model.train()
    total_loss = 0.0
    total_consensus = 0.0
    total_target = 0.0
    total_mse = 0.0
    total_ssim = 0.0

    # Get blended loss weights for this epoch
    mse_blend, ssim_blend = loss_scheduler.get_weights(epoch)

    for batch_data in loader:
        batch_x, batch_y, _, _ = batch_data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        bs = batch_x.size(0)

        optimizer.zero_grad()

        # Random diffusion timestep
        t = torch.randint(0, consensus_scheduler.T + 1, (bs,), device=device).float()
        noise = torch.randn_like(batch_y)
        noise_scales = torch.tensor([consensus_scheduler.get_noise_scale(ti.item())
                                    for ti in t], device=device).view(bs, 1)
        noisy_target = batch_y * noise_scales + noise * (1 - noise_scales)

        # Forward pass through both networks
        lin_v, lin_f = model.linear_net(batch_x, return_faces=True)
        non_v, non_f = model.nonlinear_net(batch_x, return_faces=True)

        # Z-coupling
        coupled_l = [model.linear_to_nonlinear[i](lin_f[:, i], non_f[:, i])
                     for i in range(4)]
        coupled_n = [model.nonlinear_to_linear[i](non_f[:, i], lin_f[:, i])
                     for i in range(4)]

        lin_v = model.linear_net.update_from_faces(lin_v, [f * 0.5 for f in coupled_l])
        non_v = model.nonlinear_net.update_from_faces(non_v, [f * 0.5 for f in coupled_n])

        # Separate outputs
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

        # Diffusion consensus weights
        avg_t = t.mean().item()
        consensus_weight, target_weight = consensus_scheduler.get_weights(avg_t)

        # === BLENDED LOSS ===
        # Consensus: X and Y must agree
        if ssim_blend > 0:
            consensus_loss = ssim_loss(linear_output, nonlinear_output, img_size)
        else:
            consensus_loss = F.mse_loss(linear_output, nonlinear_output)

        # Target: Both move toward target
        mse_target = (F.mse_loss(linear_output, batch_y) +
                     F.mse_loss(nonlinear_output, batch_y)) / 2

        if ssim_blend > 0:
            ssim_target = (ssim_loss(linear_output, batch_y, img_size) +
                          ssim_loss(nonlinear_output, batch_y, img_size)) / 2
            target_loss = mse_blend * mse_target + ssim_blend * ssim_target
        else:
            target_loss = mse_target

        # Combined
        loss = consensus_weight * consensus_loss + target_weight * target_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_consensus += consensus_loss.item()
        total_target += target_loss.item()
        total_mse += mse_target.item()
        if ssim_blend > 0:
            total_ssim += ssim_target.item()

    n = len(loader)
    return {
        'loss': total_loss / n,
        'consensus': total_consensus / n,
        'target': total_target / n,
        'mse': total_mse / n,
        'ssim': total_ssim / n if ssim_blend > 0 else 0.0,
        'mse_blend': mse_blend,
        'ssim_blend': ssim_blend
    }


def evaluate(model, loader, device, img_size=128):
    """Evaluate with SSIM."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_data in loader:
            batch_x, batch_y, _, _ = batch_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            total_loss += ssim_loss(output, batch_y, img_size).item()

    return total_loss / len(loader)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_nested_training(input_folder, output_folder,
                       img_size=128, latent_dim=128,
                       epochs=300, test_idx=6,
                       structured_ratio=0.2,
                       diffusion_steps=10,
                       bootstrap_epochs=30,
                       base_lr=0.0001,
                       timescale_factor=2.0,
                       load_checkpoint_path: Optional[str] = None,
                       device=None):
    """
    Run nested timescale training with blended MSE→SSIM.

    Args:
        timescale_factor: Learning rate divisor for hierarchy (e.g., 2.0 means each level is 2× slower)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print("🌀 TRAINING_10: NESTED TIMESCALES + BLENDED LOSS")
    print("="*70)
    print(f"Image size: {img_size}×{img_size}")
    print(f"Latent dim: {latent_dim}")
    print(f"Structured ratio: {structured_ratio} (Pareto)")
    print(f"Diffusion steps: {diffusion_steps}")
    print(f"Bootstrap epochs: {bootstrap_epochs} (pure MSE)")
    print(f"Total epochs: {epochs}")
    print(f"Base LR: {base_lr}")
    print(f"Timescale factor: {timescale_factor}")
    print(f"Device: {device}")
    print("\n🧠 Nested Timescales (Morphogenetic Order):")
    print(f"  Vertices:   LR = {base_lr:.6f} (fast/reactive)")
    print(f"  Edges:      LR = {base_lr/timescale_factor:.6f} (medium)")
    print(f"  Faces:      LR = {base_lr/(timescale_factor**2):.6f} (slow)")
    print(f"  Coupling:   LR = {base_lr/(timescale_factor**3):.6f} (slowest/field)")
    print("\n📊 Blended Loss Schedule:")
    print(f"  Epoch 0-{bootstrap_epochs}: 100% MSE, 0% SSIM (bootstrap)")
    print(f"  Epoch {bootstrap_epochs}-{epochs}: 100%→20% MSE, 0%→80% SSIM (blend)")
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

    # Load checkpoint if provided
    if load_checkpoint_path:
        print(f"Loading model checkpoint from {load_checkpoint_path}...")
        model.load_state_dict(torch.load(load_checkpoint_path, map_location=device))
        print("✓ Model checkpoint loaded successfully.\n")

    # === NESTED TIMESCALE OPTIMIZER ===
    # Group parameters by geometric hierarchy
    vertex_params = []
    edge_params = []
    face_params = []
    coupling_params = []

    # Embedding layers (vertices)
    vertex_params.extend(model.linear_net.embed.parameters())
    vertex_params.extend(model.nonlinear_net.embed.parameters())

    # Edge modules
    edge_params.extend(model.linear_net.edge_modules.parameters())
    edge_params.extend(model.nonlinear_net.edge_modules.parameters())

    # Face modules
    face_params.extend(model.linear_net.face_modules.parameters())
    face_params.extend(model.nonlinear_net.face_modules.parameters())

    # Inter-face coupling
    coupling_params.extend(model.linear_to_nonlinear.parameters())
    coupling_params.extend(model.nonlinear_to_linear.parameters())

    # Output projection (vertex-level)
    vertex_params.extend(model.output_projection.parameters())

    # Create optimizer with nested learning rates
    optimizer = optim.Adam([
        {'params': vertex_params, 'lr': base_lr},                           # Fast
        {'params': edge_params, 'lr': base_lr / timescale_factor},         # Medium
        {'params': face_params, 'lr': base_lr / (timescale_factor ** 2)},  # Slow
        {'params': coupling_params, 'lr': base_lr / (timescale_factor ** 3)} # Slowest
    ])

    consensus_scheduler = ConsensusScheduler(diffusion_steps=diffusion_steps)
    loss_scheduler = BlendedLossScheduler(bootstrap_epochs=bootstrap_epochs, total_epochs=epochs)

    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Parameter groups: 4 (vertices, edges, faces, coupling)")
    print(f"   Diffusion scheduler: {diffusion_steps} steps")
    print(f"   Loss scheduler: MSE→SSIM blend\n")

    # Train
    history = {
        'train': [], 'test': [],
        'consensus': [], 'target': [],
        'mse': [], 'ssim': [],
        'mse_blend': [], 'ssim_blend': []
    }

    print("Training with nested timescales + blended loss...")
    print("="*70)

    for epoch in range(epochs):
        metrics = train_nested_consensus(
            model, train_loader, optimizer, device, epoch,
            latent_dim=latent_dim,
            img_size=img_size,
            consensus_scheduler=consensus_scheduler,
            loss_scheduler=loss_scheduler
        )

        test_loss = evaluate(model, test_loader, device, img_size=img_size)

        history['train'].append(metrics['loss'])
        history['test'].append(test_loss)
        history['consensus'].append(metrics['consensus'])
        history['target'].append(metrics['target'])
        history['mse'].append(metrics['mse'])
        history['ssim'].append(metrics['ssim'])
        history['mse_blend'].append(metrics['mse_blend'])
        history['ssim_blend'].append(metrics['ssim_blend'])

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == bootstrap_epochs:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {metrics['loss']:.6f} | "
                  f"Test: {test_loss:.6f} | "
                  f"Consensus: {metrics['consensus']:.6f} | "
                  f"MSE: {metrics['mse']:.6f} ({metrics['mse_blend']*100:.0f}%) | "
                  f"SSIM: {metrics['ssim']:.6f} ({metrics['ssim_blend']*100:.0f}%)")

    print("\n✅ Training complete!\n")

    # Save model
    save_dir = Path("/content/drive/MyDrive/saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "nested_timescales_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}\n")

    # Visualize
    visualize_nested_results(model, train_dataset, history, img_size, device)

    return model, history


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_nested_results(model, train_dataset, history, img_size, device):
    """Visualize training results."""
    model.eval()

    # Get sample
    sample_x, sample_y = train_dataset.base_pairs[0]
    sample_x_tensor = sample_x.reshape(1, -1).to(device)

    with torch.no_grad():
        sample_pred = model(sample_x_tensor)

    def to_img(t):
        return t.reshape(3, img_size, img_size).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    input_img = to_img(sample_x)
    target_img = to_img(sample_y)
    pred_img = to_img(sample_pred)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: images
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_img)
    axes[0, 1].set_title('Prediction (Nested)', fontsize=14, fontweight='bold', color='blue')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(target_img)
    axes[0, 2].set_title('Target', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Bottom row: training curves
    axes[1, 0].plot(history['test'], label='Test Loss', linewidth=2, color='blue')
    axes[1, 0].set_title('Test Loss (SSIM)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history['mse_blend'], label='MSE Weight', linewidth=2, color='orange')
    axes[1, 1].plot(history['ssim_blend'], label='SSIM Weight', linewidth=2, color='purple')
    axes[1, 1].set_title('Blended Loss Weights', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(history['consensus'], label='Consensus', linewidth=2, color='green')
    axes[1, 2].plot(history['target'], label='Target', linewidth=2, color='red')
    axes[1, 2].set_title('Consensus vs Target Loss', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('nested_timescales_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'nested_timescales_results.png'\n")
    plt.show()

    # Stats
    print("="*70)
    print("📊 NESTED TIMESCALES RESULTS")
    print("="*70)
    print(f"Test Loss (SSIM):    {history['test'][-1]:.6f}")
    print(f"Consensus Loss:      {history['consensus'][-1]:.6f}")
    print(f"Target Loss:         {history['target'][-1]:.6f}")
    print(f"Final MSE component: {history['mse'][-1]:.6f}")
    print(f"Final SSIM component:{history['ssim'][-1]:.6f}")
    print("\n💡 Architecture:")
    print("  • Morphogenetic hierarchy: Field (coupling) slowest, activity (vertices) fastest")
    print("  • Blended loss: MSE bootstrap → SSIM refinement")
    print("  • Self-organized specialization: X and Y negotiate roles")
    print("="*70)


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    model, history = run_nested_training(
        input_folder="/content/drive/MyDrive/trainingdata/images_49/set_04",
        output_folder="/content/drive/MyDrive/trainingdata/images_49/set_05",
        img_size=128,
        latent_dim=128,
        epochs=300,
        test_idx=6,
        structured_ratio=0.2,      # Pareto batching
        diffusion_steps=10,
        bootstrap_epochs=30,        # Pure MSE for 30 epochs
        base_lr=0.0001,
        timescale_factor=2.0,      # Each level 2× slower
        load_checkpoint_path=None   # Or path to continue training
    )
