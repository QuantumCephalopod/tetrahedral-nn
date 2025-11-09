"""
ACTIVE INFERENCE - LIVE VISUALIZATION
======================================

Watch the world model learn in real-time!

Shows:
  - Current frame
  - Model's prediction of next frame
  - Actual next frame
  - Prediction error visualization

Can save frames for video compilation.

Author: Philipp Remy Bartholomäus
Date: November 9, 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from pathlib import Path
import time


# ============================================================================
# LIVE TRAINING WITH VISUALIZATION
# ============================================================================

def visualize_world_model_live(
    trainer,
    n_steps=100,
    update_every=10,
    save_frames=False,
    frame_dir='./frames',
    save_checkpoint_every=50
):
    """
    Train while visualizing predictions in real-time.

    Args:
        trainer: ActiveInferenceTrainer instance
        n_steps: Number of steps to run
        update_every: Update visualization every N steps
        save_frames: Save frames to disk for video creation
        frame_dir: Directory to save frames
        save_checkpoint_every: Save checkpoint every N steps (0 to disable)
    """
    if save_frames:
        frame_path = Path(frame_dir)
        frame_path.mkdir(exist_ok=True)
        print(f"💾 Saving frames to {frame_path}")

    print("\n" + "="*70)
    print("🎬 LIVE WORLD MODEL VISUALIZATION")
    print("="*70)
    print(f"Running {n_steps} steps, updating every {update_every}")
    print("="*70 + "\n")

    trainer.model.eval()

    # Get initial state
    frame, _ = trainer.env.reset()
    from ACTIVE_INFERENCE_ATARI import preprocess_frame
    frame = preprocess_frame(frame, trainer.img_size)

    frame_count = 0

    for step in range(n_steps):
        # Random action
        action = trainer.policy.select_action(frame)

        # Model prediction
        with torch.no_grad():
            frame_batch = frame.unsqueeze(0).to(trainer.device)
            action_batch = torch.tensor([action], dtype=torch.long).to(trainer.device)
            predicted_next = trainer.model(frame_batch, action_batch)
            predicted_next = predicted_next.squeeze(0).cpu()

        # Take action in environment
        next_frame_raw, reward, terminated, truncated, info = trainer.env.step(action)
        next_frame = preprocess_frame(next_frame_raw, trainer.img_size)

        # Store experience
        trainer.buffer.add(frame, action, next_frame)

        # Train on batch
        metrics = None  # Initialize to avoid UnboundLocalError
        if len(trainer.buffer) >= trainer.batch_size:
            metrics = trainer.train_step()

            # Auto-save checkpoint
            if save_checkpoint_every > 0 and trainer.step_count % save_checkpoint_every == 0:
                trainer.save_checkpoint()

        # Visualize every N steps
        if step % update_every == 0:
            clear_output(wait=True)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Current frame
            axes[0].imshow(frame.permute(1, 2, 0).numpy())
            axes[0].set_title(f'Current Frame\nStep {step}\nAction: {action}',
                            fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Predicted next
            pred_img = predicted_next.permute(1, 2, 0).numpy().clip(0, 1)
            axes[1].imshow(pred_img)
            axes[1].set_title('Predicted Next\n(World Model)',
                            fontsize=12, fontweight='bold', color='blue')
            axes[1].axis('off')

            # Actual next
            axes[2].imshow(next_frame.permute(1, 2, 0).numpy())
            axes[2].set_title('Actual Next\n(Reality)',
                            fontsize=12, fontweight='bold', color='green')
            axes[2].axis('off')

            # Prediction error (absolute difference)
            error = torch.abs(predicted_next - next_frame)
            error_img = error.permute(1, 2, 0).numpy()
            im = axes[3].imshow(error_img, cmap='hot')
            axes[3].set_title(f'Prediction Error\nMSE: {metrics["mse"]:.6f}' if metrics else 'Prediction Error',
                            fontsize=12, fontweight='bold', color='red')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3])

            plt.tight_layout()

            # Save frame
            if save_frames:
                plt.savefig(frame_path / f'frame_{frame_count:04d}.png',
                           dpi=100, bbox_inches='tight')
                frame_count += 1

            plt.show()

            # Print metrics
            if metrics:
                print(f"\n📊 Step {step}/{n_steps}")
                print(f"   Training steps: {trainer.step_count}")
                print(f"   🌱 Curriculum: {metrics['phase']}")
                print(f"   👁️  Attention: {metrics['mask_amount']*100:.1f}% masked")
                print(f"   🔄 Mode: {'Difference' if metrics.get('difference_mode', False) else 'State'}")
                print(f"   MSE Loss: {metrics['mse']:.6f}")
                print(f"   SSIM Loss: {metrics['ssim']:.6f}")
                print(f"   Buffer size: {len(trainer.buffer)}")
                print(f"   MSE weight: {metrics['mse_weight']:.2f}")
                print(f"   SSIM weight: {metrics['ssim_weight']:.2f}")

        # Next state
        frame = next_frame

        if terminated or truncated:
            frame, _ = trainer.env.reset()
            frame = preprocess_frame(frame, trainer.img_size)
            print(f"\n🎮 Episode complete at step {step}")

    trainer.model.train()

    if save_frames:
        print(f"\n✅ Saved {frame_count} frames to {frame_path}")
        print(f"\nTo create video, run:")
        print(f"  !ffmpeg -framerate 10 -i {frame_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p world_model.mp4")

    # Final checkpoint save
    if save_checkpoint_every > 0:
        trainer.save_checkpoint()

    print("\n✅ Visualization complete!\n")


# ============================================================================
# CONTINUE TRAINING (NON-VISUAL)
# ============================================================================

def continue_training(trainer, n_episodes=5, train_steps_per_episode=50, save_checkpoint=True):
    """
    Continue training without visualization (faster).

    Use this to accumulate more experience quickly,
    then use visualize_world_model_live() to see results.

    Args:
        trainer: ActiveInferenceTrainer instance
        n_episodes: Number of episodes to run
        train_steps_per_episode: Training steps per episode
        save_checkpoint: Save checkpoint after training
    """
    print("\n" + "="*70)
    print("⚡ CONTINUE TRAINING")
    print("="*70)
    print(f"Running {n_episodes} more episodes")
    print("="*70 + "\n")

    start_episode = trainer.episode_count
    start_step = trainer.step_count

    for episode in range(n_episodes):
        # Collect experience
        episodes_done = trainer.collect_experience(n_steps=100)
        trainer.episode_count += episodes_done

        # Train
        print(f"🎓 Training for {train_steps_per_episode} steps...")
        episode_metrics = {'mse': [], 'ssim': [], 'total': []}

        for _ in range(train_steps_per_episode):
            metrics = trainer.train_step()
            if metrics:
                episode_metrics['mse'].append(metrics['mse'])
                episode_metrics['ssim'].append(metrics['ssim'])
                episode_metrics['total'].append(metrics['total'])

        # Log
        avg_mse = np.mean(episode_metrics['mse']) if episode_metrics['mse'] else 0
        avg_ssim = np.mean(episode_metrics['ssim']) if episode_metrics['ssim'] else 0
        avg_total = np.mean(episode_metrics['total']) if episode_metrics['total'] else 0

        print(f"\n📊 Episode {episode+1}/{n_episodes}")
        print(f"   Total episodes: {trainer.episode_count}")
        print(f"   Training steps: {trainer.step_count}")
        print(f"   MSE: {avg_mse:.6f}")
        print(f"   SSIM: {avg_ssim:.6f}")
        print(f"   Total: {avg_total:.6f}")
        print(f"   Buffer: {len(trainer.buffer)}")

    episodes_trained = trainer.episode_count - start_episode
    steps_trained = trainer.step_count - start_step

    print(f"\n✅ Trained {episodes_trained} episodes, {steps_trained} steps")

    # Save checkpoint
    if save_checkpoint:
        trainer.save_checkpoint()

    print("="*70 + "\n")


# ============================================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================================

def compare_predictions_grid(trainer, n_samples=6):
    """
    Show grid of predictions from buffer.

    Good for seeing overall quality of world model.
    """
    if len(trainer.buffer) < n_samples:
        print(f"Not enough samples in buffer ({len(trainer.buffer)})")
        return

    trainer.model.eval()

    frames, actions, next_frames = trainer.buffer.sample(n_samples)
    frames = frames.to(trainer.device)
    actions = actions.to(trainer.device)

    with torch.no_grad():
        predicted_next = trainer.model(frames, actions)

    # Calculate errors
    mse_errors = []
    for i in range(n_samples):
        mse = torch.mean((predicted_next[i] - next_frames[i]) ** 2).item()
        mse_errors.append(mse)

    # Plot grid
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Current
        axes[i, 0].imshow(frames[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Current\nAction: {actions[i].item()}')
        axes[i, 0].axis('off')

        # Predicted
        pred_img = predicted_next[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 1].imshow(pred_img)
        axes[i, 1].set_title('Predicted', color='blue')
        axes[i, 1].axis('off')

        # Actual
        axes[i, 2].imshow(next_frames[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Actual', color='green')
        axes[i, 2].axis('off')

        # Error
        error = torch.abs(predicted_next[i].cpu() - next_frames[i])
        error_img = error.permute(1, 2, 0).numpy()
        im = axes[i, 3].imshow(error_img, cmap='hot')
        axes[i, 3].set_title(f'Error\nMSE: {mse_errors[i]:.6f}', color='red')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('world_model_grid.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to 'world_model_grid.png'")
    plt.show()

    trainer.model.train()

    print(f"\n📊 Prediction Quality:")
    print(f"   Average MSE: {np.mean(mse_errors):.6f}")
    print(f"   Best MSE: {np.min(mse_errors):.6f}")
    print(f"   Worst MSE: {np.max(mse_errors):.6f}")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎬 LIVE VISUALIZATION TOOLS")
    print("="*70)
    print("\nAvailable functions:")
    print("\n1. visualize_world_model_live(trainer, n_steps=100)")
    print("   → Watch predictions in real-time while training")
    print("\n2. continue_training(trainer, n_episodes=5)")
    print("   → Train more episodes quickly (no visualization)")
    print("\n3. compare_predictions_grid(trainer, n_samples=6)")
    print("   → Show grid of predictions from buffer")
    print("\nExample:")
    print("  # Train with live visualization")
    print("  visualize_world_model_live(trainer, n_steps=100, save_frames=True)")
    print("\n  # Or train fast, then visualize")
    print("  continue_training(trainer, n_episodes=10)")
    print("  compare_predictions_grid(trainer, n_samples=9)")
    print("="*70 + "\n")
