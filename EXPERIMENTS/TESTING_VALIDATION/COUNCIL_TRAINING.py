"""
COUNCIL OF ADVERSARIES - COMPLETE TRAINING SCRIPT

Run this to train the council architecture on image transformation!

Usage:
    python COUNCIL_TRAINING.py

Or in Colab:
    %run COUNCIL_TRAINING.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import time
import sys

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Z_COUPLING.COUNCIL_OF_ADVERSARIES import (
    CouncilOfAdversariesNetwork,
    ExternalDiscriminator,
    train_council_step,
    visualize_council_field
)

# ============================================================================
# DATA LOADING (from your existing setup)
# ============================================================================

def create_edge_version(img):
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_3ch = np.stack([edges, edges, edges], axis=2) / 255.0
    return torch.from_numpy(edges_3ch).permute(2, 0, 1).float()

def create_grayscale_version(img):
    img_np = img.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_3ch = np.stack([gray, gray, gray], axis=2) / 255.0
    return torch.from_numpy(gray_3ch).permute(2, 0, 1).float()

def create_dithered_version(img):
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    dithered = (gray > 128).astype(np.uint8) * 255
    dithered_3ch = np.stack([dithered, dithered, dithered], axis=2) / 255.0
    return torch.from_numpy(dithered_3ch).permute(2, 0, 1).float()


class ImagePairLoader:
    """Simple image loader"""
    def __init__(self, input_folder, output_folder, target_size=128):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size

    def _load_image(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)


class MultiRepresentationDataset(Dataset):
    def __init__(self, image_pairs, apply_augmentation=True):
        self.base_pairs = []
        for input_img, output_img in image_pairs:
            if apply_augmentation:
                augs_in = self.augment_image_simple(input_img)
                augs_out = self.augment_image_simple(output_img)
                for aug_in, aug_out in zip(augs_in, augs_out):
                    self.base_pairs.append((aug_in, aug_out))
            else:
                self.base_pairs.append((input_img, output_img))
        self.n_base = len(self.base_pairs)
        print(f"  Base pairs: {self.n_base}, Total: {self.n_base * 4}")

    def augment_image_simple(self, img):
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


class HybridBatchSampler(Sampler):
    """Mix structured (teaching) and random (experience) batches"""
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

        # Structured batches (same pair, all 4 reps)
        for i in range(n_structured):
            base_idx = base_indices[i]
            batch = [base_idx * 4 + rep for rep in range(4)]
            batches.append(batch)

        # Random batches
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
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Paths (UPDATE THESE for your setup)
    'input_folder': "/content/drive/MyDrive/trainingdata/images_49/set_04",
    'output_folder': "/content/drive/MyDrive/trainingdata/images_49/set_05",

    # Architecture
    'img_size': 128,
    'latent_dim': 128,
    'num_candidates': 4,  # How many outputs each network generates
    'coupling_strength': 0.5,

    # Training
    'epochs': 500,
    'lr_generator': 0.0001,
    'lr_discriminator': 0.0001,
    'test_idx': 6,  # Which image pair to hold out for testing
    'structured_ratio': 0.2,  # 20% structured teaching, 80% random experience

    # Loss weights
    'input_recon_weight': 0.1,
    'diversity_weight': 0.5,
    'consensus_weight': 0.3,
    'adversarial_weight': 0.2,

    # Visualization
    'vis_every': 50,  # Visualize field every N epochs
    'print_every': 10,

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("="*70)
    print("🔷 COUNCIL OF ADVERSARIES TRAINING")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"  Image size: {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"  Latent dim: {CONFIG['latent_dim']}")
    print(f"  Candidates per network: {CONFIG['num_candidates']}")
    print(f"  Total candidates in field: {CONFIG['num_candidates'] * 2}")
    print(f"  Coupling strength: {CONFIG['coupling_strength']}")
    print(f"  Structured ratio: {CONFIG['structured_ratio']}")
    print(f"  Device: {CONFIG['device']}")
    print("="*70 + "\n")

    # ========================================
    # LOAD DATA
    # ========================================
    print("📂 Loading data...")
    loader = ImagePairLoader(
        CONFIG['input_folder'],
        CONFIG['output_folder'],
        target_size=CONFIG['img_size']
    )

    input_paths = sorted(Path(CONFIG['input_folder']).glob('*'))
    output_paths = sorted(Path(CONFIG['output_folder']).glob('*'))

    all_pairs = []
    for i in range(7):
        inp = loader._load_image(input_paths[i])
        out = loader._load_image(output_paths[i])
        all_pairs.append((inp, out))

    train_pairs = [p for i, p in enumerate(all_pairs) if i != CONFIG['test_idx']]
    test_pairs = [all_pairs[CONFIG['test_idx']]]

    train_dataset = MultiRepresentationDataset(train_pairs, apply_augmentation=True)
    test_dataset = MultiRepresentationDataset(test_pairs, apply_augmentation=True)

    train_sampler = HybridBatchSampler(
        train_dataset,
        structured_ratio=CONFIG['structured_ratio']
    )
    test_sampler = HybridBatchSampler(
        test_dataset,
        structured_ratio=0.0,
        shuffle=False
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}\n")

    # ========================================
    # BUILD MODEL
    # ========================================
    print("🏗️  Building Council of Adversaries...")

    input_dim = CONFIG['img_size'] * CONFIG['img_size'] * 3

    model = CouncilOfAdversariesNetwork(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=CONFIG['latent_dim'],
        num_candidates=CONFIG['num_candidates'],
        coupling_strength=CONFIG['coupling_strength']
    ).to(CONFIG['device'])

    discriminator = ExternalDiscriminator(input_dim).to(CONFIG['device'])

    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print()

    # ========================================
    # OPTIMIZERS
    # ========================================
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr_generator'], betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=CONFIG['lr_discriminator'], betas=(0.5, 0.999))

    # ========================================
    # TRAINING LOOP
    # ========================================
    print("⚡ Training...\n")
    print("Epoch | Gen Loss | Disc Loss | Realism | Diversity | Consensus | Real→Fake | Entropy")
    print("-" * 90)

    history = {
        'train_gen_loss': [],
        'train_disc_loss': [],
        'test_gen_loss': [],
        'test_disc_loss': []
    }

    for epoch in range(CONFIG['epochs']):
        model.train()
        discriminator.train()

        epoch_metrics = {
            'gen_total': 0,
            'disc_loss': 0,
            'gen_realism': 0,
            'diversity': 0,
            'consensus': 0,
            'real_score': 0,
            'fake_score': 0,
            'weight_entropy': 0
        }
        n_batches = 0

        start_time = time.time()

        # Train
        for batch_data in train_loader:
            batch_x, batch_y, _, _ = batch_data
            batch_x = batch_x.to(CONFIG['device'])
            batch_y = batch_y.to(CONFIG['device'])

            gen_loss, disc_loss, metrics = train_council_step(
                model, discriminator,
                batch_x, batch_y,
                optimizer, disc_optimizer,
                input_recon_weight=CONFIG['input_recon_weight'],
                diversity_weight=CONFIG['diversity_weight'],
                consensus_weight=CONFIG['consensus_weight'],
                adversarial_weight=CONFIG['adversarial_weight']
            )

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            n_batches += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        history['train_gen_loss'].append(epoch_metrics['gen_total'])
        history['train_disc_loss'].append(epoch_metrics['disc_loss'])

        # Test
        model.eval()
        discriminator.eval()
        test_gen_loss = 0
        test_disc_loss = 0

        with torch.no_grad():
            for batch_data in test_loader:
                batch_x, batch_y, _, _ = batch_data
                batch_x = batch_x.to(CONFIG['device'])
                batch_y = batch_y.to(CONFIG['device'])

                output, field_data = model(batch_x, return_field=True)
                recon = field_data['reconstructed_input']

                # Simple MSE for test (no GAN during eval)
                test_gen_loss += F.mse_loss(output, batch_y).item()
                test_gen_loss += CONFIG['input_recon_weight'] * F.mse_loss(recon, batch_x).item()

        test_gen_loss /= len(test_loader)
        history['test_gen_loss'].append(test_gen_loss)

        # Print
        if (epoch + 1) % CONFIG['print_every'] == 0:
            print(f"{epoch+1:5d} | {epoch_metrics['gen_total']:8.4f} | "
                  f"{epoch_metrics['disc_loss']:9.4f} | "
                  f"{epoch_metrics['gen_realism']:7.4f} | "
                  f"{epoch_metrics['diversity']:9.4f} | "
                  f"{epoch_metrics['consensus']:9.4f} | "
                  f"{epoch_metrics['real_score']:.2f}→{epoch_metrics['fake_score']:.2f} | "
                  f"{epoch_metrics['weight_entropy']:7.3f}")

        # Visualize field
        if (epoch + 1) % CONFIG['vis_every'] == 0:
            print(f"\n📸 Visualizing field at epoch {epoch+1}...")
            test_batch = next(iter(test_loader))
            test_x = test_batch[0].to(CONFIG['device'])
            test_y = test_batch[1].to(CONFIG['device'])
            visualize_council_field(model, test_x, test_y, CONFIG['device'], epoch+1)
            print("✓ Saved visualization\n")

    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)

    # Final visualization
    print("\n📸 Generating final visualization...")
    test_batch = next(iter(test_loader))
    test_x = test_batch[0].to(CONFIG['device'])
    test_y = test_batch[1].to(CONFIG['device'])
    visualize_council_field(model, test_x, test_y, CONFIG['device'], 'FINAL')

    # Plot training curves
    print("📊 Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_gen_loss'], label='Train Gen', alpha=0.7)
    axes[0].plot(history['test_gen_loss'], label='Test', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Generator Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_disc_loss'], label='Discriminator', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Discriminator Loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('council_training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training history plot\n")

    # Save model
    print("💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'config': CONFIG,
        'history': history
    }, 'council_model_final.pt')
    print("✓ Saved to council_model_final.pt\n")

    print("="*70)
    print("🎉 All done! Check the visualizations!")
    print("="*70)


if __name__ == "__main__":
    main()
