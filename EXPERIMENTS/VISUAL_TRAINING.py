"""
VISUAL TRAINING - Actually SEE what's happening!

This will show you:
- Live training plots
- Confusion matrices
- State representations
- Everything updating in real-time

No more blind flying!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import clear_output, display
import seaborn as sns


def train_with_live_visualization(trainer, n_episodes=10, steps_per_episode=50, update_every=2):
    """
    Train with LIVE visualization - see everything as it happens!
    """

    print("="*70)
    print("🎬 LIVE VISUAL TRAINING")
    print("="*70)
    print("Updating plots every", update_every, "episodes")
    print("="*70 + "\n")

    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

    for episode in range(n_episodes):
        # Collect experience
        episodes_done = trainer.collect_experience(n_steps=100)
        trainer.episode_count += episodes_done

        # Train
        episode_metrics = {'forward': [], 'inverse': [], 'accuracy': []}

        for _ in range(steps_per_episode):
            metrics = trainer.train_step()
            if metrics:
                episode_metrics['forward'].append(metrics['forward'])
                episode_metrics['inverse'].append(metrics['inverse'])
                episode_metrics['accuracy'].append(metrics['accuracy'])

        # Print progress
        if episode_metrics['forward']:
            avg_forward = np.mean(episode_metrics['forward'])
            avg_inverse = np.mean(episode_metrics['inverse'])
            avg_accuracy = np.mean(episode_metrics['accuracy'])

            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Acc={avg_accuracy*100:.1f}% "
                  f"InvLoss={avg_inverse:.3f} "
                  f"FwdLoss={avg_forward:.6f}")

        # Visualize every N episodes
        if (episode + 1) % update_every == 0 or episode == n_episodes - 1:
            clear_output(wait=True)
            visualize_everything(trainer, action_names)
            plt.show()

            print(f"\n{'='*70}")
            print(f"📊 After Episode {episode+1}/{n_episodes}")
            print(f"{'='*70}")
            print(f"Accuracy: {avg_accuracy*100:.2f}% (random = 16.67%)")
            print(f"Inverse loss: {avg_inverse:.4f} (random baseline = {np.log(6):.4f})")
            print(f"Forward loss: {avg_forward:.6f}")

            if avg_accuracy > 0.20:
                print("✅ Accuracy > 20% - Model is learning!")
            else:
                print("⚠️  Accuracy ≤ 20% - Still near random")

            if avg_inverse < 1.75:
                print("✅ Inverse loss < 1.75 - Making progress!")
            else:
                print("⚠️  Inverse loss ≥ 1.75 - Near random baseline")

            print(f"{'='*70}\n")


def visualize_everything(trainer, action_names):
    """
    Create comprehensive visualization of current training state.
    """

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ========================================================================
    # ROW 1: Training Curves
    # ========================================================================

    # Forward loss
    ax1 = fig.add_subplot(gs[0, 0])
    if len(trainer.history['forward']) > 0:
        ax1.plot(trainer.history['forward'], linewidth=2)
        ax1.set_title('Forward Loss (Prediction)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('MSE')
        ax1.grid(alpha=0.3)
        ax1.set_yscale('log')

    # Inverse loss
    ax2 = fig.add_subplot(gs[0, 1])
    if len(trainer.history['inverse']) > 0:
        ax2.plot(trainer.history['inverse'], linewidth=2, color='orange')
        ax2.axhline(np.log(6), color='red', linestyle='--', linewidth=2, label='Random baseline')
        ax2.set_title('Inverse Loss (Causality)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cross-Entropy')
        ax2.legend()
        ax2.grid(alpha=0.3)

    # Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    if len(trainer.history['accuracy']) > 0:
        acc_array = np.array(trainer.history['accuracy']) * 100
        ax3.plot(acc_array, linewidth=2, color='green')
        ax3.axhline(100/6, color='red', linestyle='--', linewidth=2, label='Random (16.67%)')
        ax3.fill_between(range(len(acc_array)), 100/6, acc_array,
                         where=(acc_array > 100/6), alpha=0.3, color='green')
        ax3.set_title('Inverse Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim([0, 100])
        ax3.legend()
        ax3.grid(alpha=0.3)

    # ========================================================================
    # ROW 2: Confusion Matrix & Action Distribution
    # ========================================================================

    # Get confusion matrix
    if len(trainer.buffer) >= 100:
        confusion, action_counts = compute_confusion_matrix(trainer, n_samples=100)

        # Confusion matrix
        ax4 = fig.add_subplot(gs[1, :2])
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=action_names, yticklabels=action_names,
                   ax=ax4, cbar_kws={'label': 'Probability'}, vmin=0, vmax=1)
        ax4.set_title('Inverse Model Confusion Matrix\n(Row=True Action, Col=Predicted)',
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Predicted Action')
        ax4.set_ylabel('True Action')

        # Check if diagonal is strong
        diagonal_strength = np.mean(np.diag(confusion))
        if diagonal_strength > 0.3:
            ax4.text(0.5, 1.05, f'✅ Diagonal strength: {diagonal_strength:.2%}',
                    transform=ax4.transAxes, ha='center', color='green', fontweight='bold')
        else:
            ax4.text(0.5, 1.05, f'⚠️ Diagonal strength: {diagonal_strength:.2%} (near random)',
                    transform=ax4.transAxes, ha='center', color='red', fontweight='bold')

        # Action distribution in buffer
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.bar(range(len(action_names)), action_counts, color='steelblue', alpha=0.7)
        ax5.axhline(1/6, color='red', linestyle='--', linewidth=2, label='Uniform')
        ax5.set_xticks(range(len(action_names)))
        ax5.set_xticklabels(action_names, rotation=45, ha='right')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Action Distribution in Buffer', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3, axis='y')

    # ========================================================================
    # ROW 3: Example Predictions
    # ========================================================================

    if len(trainer.buffer) >= 4:
        visualize_predictions(trainer, fig, gs[2, :], action_names)

    plt.suptitle(f'Training Progress - Step {trainer.step_count}',
                fontsize=16, fontweight='bold', y=0.995)


def compute_confusion_matrix(trainer, n_samples=100):
    """
    Compute confusion matrix for inverse model.
    """
    trainer.model.eval()

    frames_t, actions, frames_t1 = trainer.buffer.sample(n_samples)
    frames_t = frames_t.to(trainer.device)
    frames_t1 = frames_t1.to(trainer.device)
    actions = actions.to(trainer.device)

    with torch.no_grad():
        state_t = trainer.model.encode_state(frames_t)
        state_t1 = trainer.model.encode_state(frames_t1)
        action_logits = trainer.model.inverse_model(state_t, state_t1)
        predicted_actions = torch.argmax(action_logits, dim=-1)

    # Confusion matrix
    actions_np = actions.cpu().numpy()
    predicted_np = predicted_actions.cpu().numpy()

    confusion = np.zeros((trainer.n_actions, trainer.n_actions))
    for true, pred in zip(actions_np, predicted_np):
        confusion[true, pred] += 1

    # Normalize by row
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion = confusion / row_sums

    # Action distribution
    action_counts = np.bincount(actions_np, minlength=trainer.n_actions) / len(actions_np)

    trainer.model.train()

    return confusion, action_counts


def visualize_predictions(trainer, fig, gs_section, action_names):
    """
    Visualize example predictions from forward and inverse models.
    """
    trainer.model.eval()

    frames_t, actions, frames_t1 = trainer.buffer.sample(3)
    frames_t = frames_t.to(trainer.device)
    frames_t1 = frames_t1.to(trainer.device)
    actions = actions.to(trainer.device)

    with torch.no_grad():
        # Encode states
        state_t = trainer.model.encode_state(frames_t)
        state_t1_actual = trainer.model.encode_state(frames_t1)

        # Forward prediction
        state_t1_pred = trainer.model.forward_model(state_t, actions)

        # Inverse prediction
        action_logits = trainer.model.inverse_model(state_t, state_t1_actual)
        action_pred = torch.argmax(action_logits, dim=-1)
        action_probs = torch.softmax(action_logits, dim=-1)

    # Create subplots for examples
    for i in range(3):
        ax = fig.add_subplot(gs_section[i])

        # Show frame pair side by side
        frame_pair = torch.cat([frames_t[i], frames_t1[i]], dim=2)  # Horizontal concat
        ax.imshow(frame_pair.cpu().permute(1, 2, 0).numpy())

        # Overlay info
        true_action = action_names[actions[i]]
        pred_action = action_names[action_pred[i]]
        confidence = action_probs[i, action_pred[i]].item()

        # Forward error
        fwd_error = torch.nn.functional.mse_loss(state_t1_pred[i], state_t1_actual[i]).item()

        # Color code based on correctness
        color = 'green' if true_action == pred_action else 'red'

        title = f"True: {true_action} | Pred: {pred_action} ({confidence:.1%})\nForward MSE: {fwd_error:.6f}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.axis('off')

    trainer.model.train()


# ============================================================================
# SIMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    VISUAL TRAINING
    ===============

    Usage:

    # After loading W, X, Y, Z
    from MOTION_INVERSE_FIXED import FixedTrainer
    from VISUAL_TRAINING import train_with_live_visualization

    trainer = FixedTrainer()
    train_with_live_visualization(trainer, n_episodes=10, update_every=2)

    This will show you:
    - Training curves (forward, inverse, accuracy)
    - Confusion matrix (is inverse learning?)
    - Example predictions
    - All updating live!
    """)
