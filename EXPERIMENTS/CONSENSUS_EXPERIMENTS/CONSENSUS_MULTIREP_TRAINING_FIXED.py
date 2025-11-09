# ============================================================================
# QUICK FIX FOR COLAB: Simplified Visualization and Inference
# ============================================================================
# Copy this cell and run it in Colab after the main experiment errors out

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def simple_visualize_and_infer(model_mse, model_consensus, train_dataset,
                                img_size, device, n_samples=4):
    """
    Simplified visualization that avoids shape issues.
    Shows MSE vs Consensus outputs side-by-side.
    """

    print("\n" + "="*70)
    print("📊 SIMPLIFIED VISUALIZATION & INFERENCE")
    print("="*70)

    model_mse.eval()
    model_consensus.eval()

    # Get samples
    n_samples = min(n_samples, len(train_dataset.base_pairs))

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    def to_img(tensor):
        img = tensor.reshape(3, img_size, img_size).permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)

    all_mse_losses = []
    all_cons_losses = []

    with torch.no_grad():
        for idx in range(n_samples):
            # Get sample
            test_input = train_dataset.base_pairs[idx][0]
            test_target = train_dataset.base_pairs[idx][1]
            test_input_tensor = test_input.unsqueeze(0).to(device)
            test_target_tensor = test_target.unsqueeze(0).to(device)

            # Simple forward pass - no fancy perspective extraction
            mse_output = model_mse(test_input_tensor).cpu()
            cons_output = model_consensus(test_input_tensor).cpu()

            # Calculate losses
            mse_loss = F.mse_loss(mse_output, test_target_tensor.cpu()).item()
            cons_loss = F.mse_loss(cons_output, test_target_tensor.cpu()).item()

            all_mse_losses.append(mse_loss)
            all_cons_losses.append(cons_loss)

            # Convert to images
            input_img = to_img(test_input)
            target_img = to_img(test_target)
            mse_img = to_img(mse_output)
            cons_img = to_img(cons_output)

            # Calculate sharpness (variance)
            mse_sharp = mse_img.var()
            cons_sharp = cons_img.var()

            # Plot
            axes[idx, 0].imshow(input_img)
            axes[idx, 0].set_title(f'Input {idx+1}', fontsize=11, fontweight='bold')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(target_img)
            axes[idx, 1].set_title(f'Target {idx+1}', fontsize=11, fontweight='bold')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(mse_img)
            axes[idx, 2].set_title(f'MSE Output\nLoss: {mse_loss:.4f}\nVar: {mse_sharp:.4f}',
                                   fontsize=10, color='red')
            axes[idx, 2].axis('off')

            axes[idx, 3].imshow(cons_img)
            axes[idx, 3].set_title(f'Consensus Output\nLoss: {cons_loss:.4f}\nVar: {cons_sharp:.4f}',
                                   fontsize=10, color='blue')
            axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig('simple_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved to 'simple_comparison.png'")
    plt.show()

    # Print statistics
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)

    mse_mean = np.mean(all_mse_losses)
    cons_mean = np.mean(all_cons_losses)

    print(f"\nAverage Loss Across {n_samples} Samples:")
    print(f"  MSE:       {mse_mean:.6f}")
    print(f"  Consensus: {cons_mean:.6f}")

    improvement = ((mse_mean - cons_mean) / mse_mean) * 100
    if cons_mean < mse_mean:
        print(f"\n✓ Consensus is {improvement:.1f}% better on average!")
    else:
        print(f"\n→ MSE is {-improvement:.1f}% better on average")

    print("\n💭 VISUAL INSPECTION:")
    print("  • Which outputs look sharper?")
    print("  • Which captures the transformation better?")
    print("  • Does one look 'mushy' or averaged?")
    print("  • Trust your eyes, not just the numbers!")

    print("\n" + "="*70)


# ============================================================================
# USAGE IN COLAB
# ============================================================================

# After your training completes (even if visualization errors out),
# the models should still be trained. Run this:

# simple_visualize_and_infer(
#     model_mse=model_mse,
#     model_consensus=model_consensus,
#     train_dataset=train_dataset,
#     img_size=128,
#     device=device,
#     n_samples=4
# )
