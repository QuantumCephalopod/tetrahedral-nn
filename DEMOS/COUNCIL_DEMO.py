"""
Quick demo of Council of Adversaries architecture.
Shows field generation and voting without full training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Z_COUPLING.COUNCIL_OF_ADVERSARIES import CouncilOfAdversariesNetwork

def demo_council():
    print("="*70)
    print("🔷 COUNCIL OF ADVERSARIES - QUICK DEMO")
    print("="*70)

    # Simple config
    img_size = 64  # Small for demo
    input_dim = img_size * img_size * 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n📊 Config:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Input dim: {input_dim}")
    print(f"  Device: {device}")
    print(f"  Candidates: 4 per network (8 total)")

    # Create model
    print("\n🏗️  Building model...")
    model = CouncilOfAdversariesNetwork(
        input_dim=input_dim,
        output_dim=input_dim,
        latent_dim=64,
        num_candidates=4,
        coupling_strength=0.5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")

    # Create random input (like a noisy image)
    print("\n🎲 Creating random input...")
    batch_size = 2
    random_input = torch.randn(batch_size, input_dim).to(device) * 0.5 + 0.5
    random_input = random_input.clamp(0, 1)

    # Forward pass WITH field
    print("\n⚡ Running forward pass (with field)...")
    model.eval()
    with torch.no_grad():
        output, field_data = model(random_input, return_field=True)

    # Unpack field data
    lin_field = field_data['linear_field']
    non_field = field_data['nonlinear_field']
    all_candidates = field_data['all_candidates']
    lin_judgments = field_data['linear_judgments']
    non_judgments = field_data['nonlinear_judgments']
    weights = field_data['consensus_weights']

    print(f"✓ Generated field!")
    print(f"\n📊 Field Structure:")
    print(f"  Linear candidates shape: {lin_field.shape}")
    print(f"  Nonlinear candidates shape: {non_field.shape}")
    print(f"  All candidates shape: {all_candidates.shape}")
    print(f"  Linear judgments shape: {lin_judgments.shape}")
    print(f"  Nonlinear judgments shape: {non_judgments.shape}")
    print(f"  Consensus weights shape: {weights.shape}")

    # Analyze first sample
    print(f"\n🔍 Analyzing Sample 0:")
    print(f"\n  Consensus Weights:")
    for i in range(4):
        print(f"    Linear-{i}: {weights[0, i]:.4f}")
    for i in range(4):
        print(f"    Nonlinear-{i}: {weights[0, i+4]:.4f}")

    # Entropy (how spread out are the weights?)
    entropy = -(weights[0] * torch.log(weights[0] + 1e-8)).sum().item()
    max_entropy = np.log(8)  # 8 candidates
    print(f"\n  Weight Entropy: {entropy:.3f} / {max_entropy:.3f}")
    print(f"  Normalized: {entropy/max_entropy:.1%}")

    if entropy < 1.0:
        print(f"  → Low entropy: Single candidate dominates (collapsed)")
    elif entropy < 1.5:
        print(f"  → Medium entropy: Few candidates preferred")
    else:
        print(f"  → High entropy: Democratic voting (good!)")

    # Face judgments
    print(f"\n  Face Judgments (Sample 0):")
    print(f"    Linear faces voting on all candidates:")
    for face_idx in range(4):
        votes = lin_judgments[0, face_idx].cpu().numpy()
        print(f"      Face-{face_idx}: {' '.join([f'{v:.2f}' for v in votes])}")

    print(f"\n    Nonlinear faces voting on all candidates:")
    for face_idx in range(4):
        votes = non_judgments[0, face_idx].cpu().numpy()
        print(f"      Face-{face_idx}: {' '.join([f'{v:.2f}' for v in votes])}")

    # Face agreement
    lin_variance = lin_judgments[0].var(dim=0).mean().item()
    non_variance = non_judgments[0].var(dim=0).mean().item()
    print(f"\n  Face Agreement:")
    print(f"    Linear faces variance: {lin_variance:.4f}")
    print(f"    Nonlinear faces variance: {non_variance:.4f}")
    if lin_variance < 0.01:
        print(f"    → Linear faces in strong agreement")
    elif lin_variance < 0.05:
        print(f"    → Linear faces mostly agree")
    else:
        print(f"    → Linear faces have different opinions")

    # Visualize
    print(f"\n🎨 Creating visualization...")

    fig, axes = plt.subplots(3, 9, figsize=(18, 6))

    # Sample 0
    sample_idx = 0

    # Row 0: Linear candidates
    for i in range(4):
        img = lin_field[sample_idx, i].cpu().reshape(img_size, img_size, 3).numpy()
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].set_title(f'Lin-{i}\nw={weights[sample_idx, i]:.3f}', fontsize=8)
        axes[0, i].axis('off')

    # Row 1: Nonlinear candidates
    for i in range(4):
        img = non_field[sample_idx, i].cpu().reshape(img_size, img_size, 3).numpy()
        axes[1, i].imshow(np.clip(img, 0, 1))
        axes[1, i].set_title(f'Non-{i}\nw={weights[sample_idx, i+4]:.3f}', fontsize=8)
        axes[1, i].axis('off')

    # Row 2: Input and consensus
    input_img = random_input[sample_idx].cpu().reshape(img_size, img_size, 3).numpy()
    output_img = output[sample_idx].cpu().reshape(img_size, img_size, 3).numpy()

    axes[2, 0].imshow(np.clip(input_img, 0, 1))
    axes[2, 0].set_title('Input', fontsize=8)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(np.clip(output_img, 0, 1))
    axes[2, 1].set_title('Consensus', fontsize=8)
    axes[2, 1].axis('off')

    # Hide unused
    for i in range(4, 9):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    for i in range(2, 9):
        axes[2, i].axis('off')

    plt.suptitle(f'Council Field (untrained) - Entropy: {entropy:.2f}', fontsize=12)
    plt.tight_layout()
    plt.savefig('council_demo_field.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to council_demo_field.png")

    # Compare standard forward vs field
    print(f"\n🔄 Comparing modes...")
    with torch.no_grad():
        simple_output, recon = model(random_input, return_field=False)

    diff = (output - simple_output).abs().mean().item()
    print(f"  Difference between field mode and simple mode: {diff:.6f}")
    print(f"  (Should be near 0 - same computation path)")

    print("\n" + "="*70)
    print("✅ Demo complete!")
    print("="*70)
    print("\nKey observations:")
    print("  1. Each network generates 4 different candidates")
    print("  2. All 8 faces vote on all 8 candidates")
    print("  3. Weights show which candidates are preferred")
    print("  4. Consensus output is weighted combination")
    print("\nNote: This is untrained! Outputs are random.")
    print("After training, candidates will be realistic and diverse.")
    print("="*70)

if __name__ == "__main__":
    demo_council()
