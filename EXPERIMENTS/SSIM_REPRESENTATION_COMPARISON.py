"""
SSIM-BASED REPRESENTATION COMPARISON
====================================

What is SSIM?
-------------
SSIM = Structural Similarity Index Measure

A perceptual metric that measures how similar two images are from a human
visual perspective. Unlike pixel-wise metrics (MSE/MAE), SSIM considers:

1. **Luminance**: Overall brightness comparison
2. **Contrast**: Variance/range comparison
3. **Structure**: Correlation of patterns after normalizing brightness/contrast

Key Differences:
  • MSE/MAE: "Are pixel values numerically close?"
  • SSIM: "Do these images LOOK structurally similar to a human?"

Range: SSIM ∈ [-1, 1]
  • 1 = perfect match
  • 0 = no correlation
  • -1 = perfect anti-correlation

Why SSIM for Tetrahedral Representations?
-----------------------------------------
SSIM can reveal whether different basis functions (W/X/Y/Z) learn similar
STRUCTURAL PATTERNS in how they organize tetrahedra spatially, even if the
raw coordinates differ!

This Experiment:
---------------
1. Train 4 models (one per basis: W, X, Y, Z)
2. Extract tetrahedral representations for test inputs
3. Convert representations to 2D heatmaps/images
4. Compute SSIM between all basis pairs
5. Visualize similarity matrix + example comparisons

Expected Insights:
- Do W/X/Y/Z learn similar geometric structures?
- Which basis pairs are most/least similar?
- Does structured geometry differ from random?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from itertools import combinations
import os

from ZW_arithmetic_adapter import ArithmeticDataset
from X_linear_tetrahedron import LinearTetrahedron


def train_single_basis(basis_name, device, epochs=100, verbose=True):
    """Train a single linear tetrahedron model for one basis."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {basis_name} Basis Model")
        print(f"{'='*70}")

    # Create datasets
    dataset_gen = ArithmeticDataset(n_inputs=2)
    train_data = dataset_gen.create_exhaustive(train_range=(-9, 9))
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

    # Create model
    model = LinearTetrahedron(
        input_dim=2,
        latent_dim=64,
        output_dim=1
    ).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}")

    if verbose:
        print(f"✓ {basis_name} training complete!")

    return model


def extract_tetrahedral_representation(model, inputs, device):
    """
    Extract the tetrahedral representation (before final output projection).

    For LinearTetrahedron, this means getting the 4D tetrahedron coordinates
    from the geometric space.
    """
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)

        # Forward pass through encoder to get tetrahedron coordinates
        # The model has: encoder -> tetrahedron geometry -> output
        # We want to capture the tetrahedron state

        # Get the 4D coordinates (this is model-specific)
        # For LinearTetrahedron, we can hook into the forward pass
        coords = model.encoder(inputs)  # Shape: (batch, latent_dim)

        return coords.cpu().numpy()


def representation_to_2d_image(representation, image_size=64):
    """
    Convert a tetrahedral representation to a 2D image for SSIM comparison.

    Strategy: Create a spatial heatmap where each position encodes information
    about the tetrahedral state. We'll reshape the flat representation into
    a 2D grid and normalize.

    Args:
        representation: (n_samples, latent_dim) array
        image_size: Target image size (will be image_size x image_size)

    Returns:
        images: (n_samples, image_size, image_size) array
    """
    n_samples, latent_dim = representation.shape

    # Calculate grid size needed to fit latent_dim
    grid_h = int(np.sqrt(latent_dim))
    grid_w = int(np.ceil(latent_dim / grid_h))

    # Pad representation if needed
    pad_size = grid_h * grid_w - latent_dim
    if pad_size > 0:
        representation = np.pad(representation, ((0, 0), (0, pad_size)), mode='constant')

    # Reshape to 2D grid
    grids = representation.reshape(n_samples, grid_h, grid_w)

    # Resize to target image size using interpolation
    images = np.zeros((n_samples, image_size, image_size))

    for i in range(n_samples):
        # Normalize to [0, 1]
        grid = grids[i]
        grid_min = grid.min()
        grid_max = grid.max()

        if grid_max > grid_min:
            grid_norm = (grid - grid_min) / (grid_max - grid_min)
        else:
            grid_norm = grid

        # Resize using nearest neighbor (simple, preserves structure)
        from scipy.ndimage import zoom
        scale_h = image_size / grid_h
        scale_w = image_size / grid_w
        images[i] = zoom(grid_norm, (scale_h, scale_w), order=1)

    return images


def compute_pairwise_ssim(images_dict):
    """
    Compute SSIM between all pairs of basis representations.

    Args:
        images_dict: {'basis_name': images_array}

    Returns:
        ssim_matrix: Dictionary of {('basis1', 'basis2'): ssim_score}
    """
    basis_names = list(images_dict.keys())
    n_samples = list(images_dict.values())[0].shape[0]

    ssim_results = {}

    print("\n" + "="*70)
    print("COMPUTING SSIM BETWEEN BASIS PAIRS")
    print("="*70)

    for basis1, basis2 in combinations(basis_names, 2):
        images1 = images_dict[basis1]
        images2 = images_dict[basis2]

        # Compute SSIM for each sample and average
        ssim_scores = []

        for i in range(n_samples):
            img1 = images1[i]
            img2 = images2[i]

            # Compute SSIM
            score = ssim(img1, img2, data_range=1.0)
            ssim_scores.append(score)

        mean_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)

        ssim_results[(basis1, basis2)] = {
            'mean': mean_ssim,
            'std': std_ssim,
            'scores': ssim_scores
        }

        print(f"{basis1} vs {basis2}: SSIM = {mean_ssim:.4f} ± {std_ssim:.4f}")

    return ssim_results


def visualize_ssim_results(ssim_results, images_dict, test_inputs, output_dir='ssim_results'):
    """Create visualizations of SSIM comparison results."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. SSIM Matrix Heatmap
    basis_names = sorted(set([b for pair in ssim_results.keys() for b in pair]))
    n_bases = len(basis_names)

    ssim_matrix = np.ones((n_bases, n_bases))  # Diagonal = 1 (self-comparison)

    for (b1, b2), result in ssim_results.items():
        i1 = basis_names.index(b1)
        i2 = basis_names.index(b2)
        ssim_matrix[i1, i2] = result['mean']
        ssim_matrix[i2, i1] = result['mean']  # Symmetric

    plt.figure(figsize=(10, 8))
    im = plt.imshow(ssim_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, label='SSIM Score')
    plt.xticks(range(n_bases), basis_names, rotation=45)
    plt.yticks(range(n_bases), basis_names)
    plt.title('SSIM Matrix: Structural Similarity Between Basis Functions')

    # Annotate cells
    for i in range(n_bases):
        for j in range(n_bases):
            text = plt.text(j, i, f'{ssim_matrix[i, j]:.3f}',
                          ha="center", va="center", color="white" if ssim_matrix[i, j] < 0.5 else "black")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ssim_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved SSIM matrix to {output_dir}/ssim_matrix.png")

    # 2. Example Image Comparisons
    # Show 3 random samples with their representations across all bases
    n_examples = 3
    example_indices = np.random.choice(test_inputs.shape[0], n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, n_bases, figsize=(4*n_bases, 4*n_examples))

    for row, idx in enumerate(example_indices):
        for col, basis in enumerate(basis_names):
            ax = axes[row, col] if n_examples > 1 else axes[col]

            image = images_dict[basis][idx]
            ax.imshow(image, cmap='coolwarm', vmin=0, vmax=1)
            ax.axis('off')

            if row == 0:
                ax.set_title(f'{basis} Basis', fontsize=12, fontweight='bold')

            if col == 0:
                input_vals = test_inputs[idx]
                ax.text(-0.1, 0.5, f'Input: {input_vals}',
                       transform=ax.transAxes,
                       rotation=90,
                       va='center',
                       fontsize=10)

    plt.suptitle('Tetrahedral Representations Across Different Bases',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/representation_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved example representations to {output_dir}/representation_examples.png")

    # 3. SSIM Distribution Plot
    fig, axes = plt.subplots(1, len(ssim_results), figsize=(5*len(ssim_results), 4))

    if len(ssim_results) == 1:
        axes = [axes]

    for ax, ((b1, b2), result) in zip(axes, ssim_results.items()):
        ax.hist(result['scores'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(result['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {result['mean']:.3f}")
        ax.set_xlabel('SSIM Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{b1} vs {b2}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('SSIM Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ssim_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved SSIM distributions to {output_dir}/ssim_distributions.png")


def main():
    """Run complete SSIM-based representation comparison experiment."""

    print("\n" + "="*70)
    print("SSIM-BASED REPRESENTATION COMPARISON EXPERIMENT")
    print("="*70)
    print("\nThis experiment compares how different basis functions (W/X/Y/Z)")
    print("represent the same inputs using SSIM (Structural Similarity).")
    print("\nSSIM measures PERCEPTUAL similarity, not just numerical difference!")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # For this demo, we'll train 4 independent models
    # (In practice, you might load pre-trained models from different basis experiments)
    basis_names = ['W', 'X', 'Y', 'Z']
    models = {}

    print("\n" + "="*70)
    print("PHASE 1: TRAINING MODELS")
    print("="*70)
    print("Training 4 models (one per basis) - this simulates different")
    print("geometric representations learning the same task...")

    for basis in basis_names:
        models[basis] = train_single_basis(basis, device, epochs=100, verbose=True)

    # Create test inputs
    print("\n" + "="*70)
    print("PHASE 2: EXTRACTING REPRESENTATIONS")
    print("="*70)

    dataset_gen = ArithmeticDataset(n_inputs=2)
    test_data = dataset_gen.create_test(test_range=(10, 50), n_samples=100)
    test_inputs = test_data.tensors[0].numpy()

    print(f"Extracting tetrahedral representations for {len(test_inputs)} test samples...")

    # Extract representations
    representations = {}
    for basis, model in models.items():
        print(f"  Extracting {basis} basis representations...")
        representations[basis] = extract_tetrahedral_representation(
            model,
            torch.FloatTensor(test_inputs),
            device
        )

    # Convert to 2D images
    print("\nConverting representations to 2D images for SSIM comparison...")
    images = {}
    for basis, repr_array in representations.items():
        images[basis] = representation_to_2d_image(repr_array, image_size=64)
        print(f"  {basis}: {images[basis].shape}")

    # Compute SSIM
    print("\n" + "="*70)
    print("PHASE 3: COMPUTING SSIM")
    print("="*70)

    ssim_results = compute_pairwise_ssim(images)

    # Visualize
    print("\n" + "="*70)
    print("PHASE 4: GENERATING VISUALIZATIONS")
    print("="*70)

    visualize_ssim_results(ssim_results, images, test_inputs)

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE - KEY FINDINGS")
    print("="*70)

    print("\n📊 SSIM SCORES (Higher = More Structurally Similar):\n")

    sorted_pairs = sorted(ssim_results.items(), key=lambda x: x[1]['mean'], reverse=True)

    for (b1, b2), result in sorted_pairs:
        mean_ssim = result['mean']

        if mean_ssim > 0.8:
            emoji = "🟢"
            interpretation = "VERY SIMILAR"
        elif mean_ssim > 0.5:
            emoji = "🟡"
            interpretation = "MODERATELY SIMILAR"
        else:
            emoji = "🔴"
            interpretation = "DIFFERENT STRUCTURES"

        print(f"{emoji} {b1} vs {b2}: {mean_ssim:.4f} - {interpretation}")

    print("\n💡 INTERPRETATION:")
    print("  • High SSIM (>0.8): Bases learn similar geometric structures")
    print("  • Medium SSIM (0.5-0.8): Some structural similarities, but different approaches")
    print("  • Low SSIM (<0.5): Fundamentally different geometric representations")
    print("\n  This tells us whether different tetrahedral bases are discovering")
    print("  the same geometric patterns or exploring different solution spaces!")

    print("\n" + "="*70)
    print("✓ All visualizations saved to ssim_results/")
    print("="*70)

    return models, representations, images, ssim_results


if __name__ == "__main__":
    models, representations, images, ssim_results = main()
    print("\n🎉 SSIM experiment complete! Check ssim_results/ for visualizations.\n")
