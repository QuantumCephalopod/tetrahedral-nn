"""
SHOW RESULTS NOW - Use this with your existing trainer!

If you have a trainer that already ran, use this to see everything.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def show_what_we_have(trainer):
    """
    Show everything from current trainer state.
    """
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC RESULTS")
    print("="*70 + "\n")

    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

    # Create big figure
    fig = plt.figure(figsize=(18, 10))

    # ========================================================================
    # 1. TRAINING CURVES
    # ========================================================================

    # Accuracy
    ax1 = plt.subplot(2, 3, 1)
    if len(trainer.history['accuracy']) > 0:
        acc_array = np.array(trainer.history['accuracy']) * 100
        ax1.plot(acc_array, linewidth=2, color='green')
        ax1.axhline(100/6, color='red', linestyle='--', linewidth=2, label='Random (16.67%)')
        ax1.fill_between(range(len(acc_array)), 100/6, acc_array,
                         where=(acc_array > 100/6), alpha=0.3, color='green')
        ax1.set_title('Inverse Model Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim([0, 50])
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Print stats
        final_acc = acc_array[-1]
        avg_acc = np.mean(acc_array[-20:]) if len(acc_array) > 20 else np.mean(acc_array)
        print(f"📊 Inverse Model Accuracy:")
        print(f"   Current: {final_acc:.2f}%")
        print(f"   Recent average: {avg_acc:.2f}%")
        print(f"   Random baseline: 16.67%")
        if avg_acc > 22:
            print(f"   ✅ LEARNING! Above random baseline")
        else:
            print(f"   ⚠️  Near random - not learning yet")
        print()

    # Inverse loss
    ax2 = plt.subplot(2, 3, 2)
    if len(trainer.history['inverse']) > 0:
        ax2.plot(trainer.history['inverse'], linewidth=2, color='orange')
        ax2.axhline(np.log(6), color='red', linestyle='--', linewidth=2, label='Random baseline')
        ax2.set_title('Inverse Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)

        final_loss = trainer.history['inverse'][-1]
        print(f"📊 Inverse Loss:")
        print(f"   Current: {final_loss:.4f}")
        print(f"   Random baseline: {np.log(6):.4f}")
        if final_loss < 1.75:
            print(f"   ✅ Below baseline - learning!")
        else:
            print(f"   ⚠️  At baseline - not learning")
        print()

    # Forward loss
    ax3 = plt.subplot(2, 3, 3)
    if len(trainer.history['forward']) > 0:
        ax3.plot(trainer.history['forward'], linewidth=2, color='blue')
        ax3.set_title('Forward Loss (MSE)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_yscale('log')
        ax3.grid(alpha=0.3)

        final_fwd = trainer.history['forward'][-1]
        print(f"📊 Forward Loss:")
        print(f"   Current: {final_fwd:.6f}")
        print()

    # ========================================================================
    # 2. CONFUSION MATRIX
    # ========================================================================

    if len(trainer.buffer) >= 50:
        print("🔬 Computing confusion matrix...")

        trainer.model.eval()

        n_samples = min(100, len(trainer.buffer))
        frames_t, actions, frames_t1 = trainer.buffer.sample(n_samples)
        frames_t = frames_t.to(trainer.device)
        frames_t1 = frames_t1.to(trainer.device)
        actions = actions.to(trainer.device)

        with torch.no_grad():
            state_t = trainer.model.encode_state(frames_t)
            state_t1 = trainer.model.encode_state(frames_t1)
            action_logits = trainer.model.inverse_model(state_t, state_t1)
            predicted_actions = torch.argmax(action_logits, dim=-1)

        # Build confusion matrix
        actions_np = actions.cpu().numpy()
        predicted_np = predicted_actions.cpu().numpy()

        confusion = np.zeros((trainer.n_actions, trainer.n_actions))
        for true, pred in zip(actions_np, predicted_np):
            confusion[true, pred] += 1

        # Normalize by row
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confusion = confusion / row_sums

        # Plot
        ax4 = plt.subplot(2, 3, (4, 6))
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=action_names, yticklabels=action_names,
                   ax=ax4, cbar_kws={'label': 'Probability'}, vmin=0, vmax=1)
        ax4.set_title('Confusion Matrix: True Action → Predicted Action', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Predicted Action')
        ax4.set_ylabel('True Action')

        # Analyze diagonal
        diagonal = np.diag(confusion)
        diagonal_strength = np.mean(diagonal)

        print(f"📊 Confusion Matrix Analysis:")
        print(f"   Diagonal strength: {diagonal_strength:.2%}")
        print(f"   Random baseline: 16.67%")
        print()
        print("   Per-action accuracy:")
        for i, name in enumerate(action_names):
            mark = "✅" if diagonal[i] > 0.2 else "⚠️ "
            print(f"     {mark} {name:10s}: {diagonal[i]*100:.1f}%")
        print()

        if diagonal_strength > 0.25:
            print("   ✅ STRONG diagonal - model understands actions!")
        elif diagonal_strength > 0.20:
            print("   🟡 WEAK diagonal - starting to learn")
        else:
            print("   ⚠️  NO diagonal - basically random")
        print()

        trainer.model.train()

    plt.tight_layout()
    plt.savefig('current_results.png', dpi=150, bbox_inches='tight')
    print("💾 Saved to: current_results.png")
    plt.show()

    print("="*70)


# Quick usage
if __name__ == "__main__":
    print("""
    Usage:

    # If you have a trainer already running:
    from SHOW_RESULTS_NOW import show_what_we_have
    show_what_we_have(trainer)

    This will:
    - Show training curves
    - Show confusion matrix
    - Tell you if it's learning or not
    - Save plot to current_results.png
    """)
