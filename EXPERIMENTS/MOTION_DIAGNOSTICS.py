"""
MOTION INVERSE MODEL - DIAGNOSTICS & VISUALIZATION
==================================================

Let's see if ANY of this actually works or if it's just "high fantasy"!

Diagnostics:
1. Motion extraction: Is it meaningful?
2. Inverse model: Does it learn causality?
3. Forward model: Does it predict correctly?
4. Consistency: Do models agree?
5. Policy: Different from random?

Author: Reality Check Edition
Date: November 10, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# For frame preprocessing
from PIL import Image


# ============================================================================
# DIAGNOSTIC TOOLS
# ============================================================================

class MotionDiagnostics:
    """
    Comprehensive diagnostics for motion-based models.

    Let's actually SEE what's happening!
    """
    def __init__(self, trainer, action_names=None):
        self.trainer = trainer
        self.model = trainer.model
        self.device = trainer.device

        # Pong action names (make it interpretable!)
        if action_names is None:
            self.action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        else:
            self.action_names = action_names

    def preprocess_frame(self, frame):
        """Helper to preprocess frame"""
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame).resize((128, 128), Image.LANCZOS)
            frame = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return frame

    # ========================================================================
    # 1. MOTION EXTRACTION VISUALIZATION
    # ========================================================================

    def visualize_motion_extraction(self, n_samples=4):
        """
        Visualize what motion extractor sees.

        Shows:
        - Frame t
        - Frame t+1
        - Extracted motion (visualized as vector field or heatmap)
        """
        print("\n" + "="*70)
        print("1️⃣  MOTION EXTRACTION DIAGNOSTIC")
        print("="*70)

        if len(self.trainer.buffer) < n_samples:
            print(f"❌ Not enough samples (need {n_samples}, have {len(self.trainer.buffer)})")
            return

        self.model.eval()

        frames_t, actions, frames_t1 = self.trainer.buffer.sample(n_samples)
        frames_t = frames_t.to(self.device)
        frames_t1 = frames_t1.to(self.device)

        with torch.no_grad():
            # Extract motion representations
            motion = self.model.extract_motion(frames_t, frames_t1)

            # Also get frame difference as baseline
            frame_diff = frames_t1 - frames_t

        # Plot
        fig = plt.figure(figsize=(16, 4*n_samples))
        gs = GridSpec(n_samples, 5, figure=fig)

        for i in range(n_samples):
            # Frame t
            ax0 = fig.add_subplot(gs[i, 0])
            ax0.imshow(frames_t[i].cpu().permute(1, 2, 0).numpy())
            ax0.set_title(f'Frame t\nAction: {self.action_names[actions[i]]}')
            ax0.axis('off')

            # Frame t+1
            ax1 = fig.add_subplot(gs[i, 1])
            ax1.imshow(frames_t1[i].cpu().permute(1, 2, 0).numpy())
            ax1.set_title('Frame t+1')
            ax1.axis('off')

            # Frame difference (baseline)
            ax2 = fig.add_subplot(gs[i, 2])
            diff_vis = frame_diff[i].cpu().abs().sum(dim=0).numpy()
            im2 = ax2.imshow(diff_vis, cmap='hot')
            ax2.set_title('Frame Diff\n(Pixel-level motion)')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            # Motion representation (learned)
            ax3 = fig.add_subplot(gs[i, 3])
            motion_vis = motion[i].cpu().numpy()
            motion_2d = motion_vis.reshape(16, 8)  # Reshape to 2D for vis
            im3 = ax3.imshow(motion_2d, cmap='viridis')
            ax3.set_title('Extracted Motion\n(Learned representation)')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)

            # Motion magnitude
            ax4 = fig.add_subplot(gs[i, 4])
            ax4.bar(range(10), motion_vis[:10])  # Show first 10 dims
            ax4.set_title('Motion Vector\n(First 10 dims)')
            ax4.set_xlabel('Dimension')
            ax4.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig('diagnostic_01_motion_extraction.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: diagnostic_01_motion_extraction.png")
        plt.show()

        self.model.train()

        # Statistics
        print(f"\n📊 Motion Statistics:")
        print(f"   Motion dim: {motion.shape[1]}")
        print(f"   Motion magnitude: {motion.norm(dim=1).mean().item():.4f} ± {motion.norm(dim=1).std().item():.4f}")
        print(f"   Motion sparsity: {(motion.abs() < 0.01).float().mean().item()*100:.1f}%")

    # ========================================================================
    # 2. INVERSE MODEL DIAGNOSTIC (Critical!)
    # ========================================================================

    def diagnose_inverse_model(self, n_samples=100):
        """
        Check if inverse model learns action causality.

        Shows:
        - Confusion matrix (predicted vs actual actions)
        - Per-action accuracy
        - Which actions it confuses
        """
        print("\n" + "="*70)
        print("2️⃣  INVERSE MODEL DIAGNOSTIC (CAUSALITY)")
        print("="*70)

        if len(self.trainer.buffer) < n_samples:
            n_samples = len(self.trainer.buffer)

        self.model.eval()

        frames_t, actions, frames_t1 = self.trainer.buffer.sample(n_samples)
        frames_t = frames_t.to(self.device)
        frames_t1 = frames_t1.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            # Extract motion
            motion_t = self.model.extract_motion(frames_t, frames_t)
            motion_t1 = self.model.extract_motion(frames_t, frames_t1)

            # Predict actions
            action_logits = self.model.inverse_model(motion_t, motion_t1)
            predicted_actions = torch.argmax(action_logits, dim=-1)

        # Move to CPU for analysis
        actions_np = actions.cpu().numpy()
        predicted_np = predicted_actions.cpu().numpy()

        # Accuracy
        accuracy = (predicted_np == actions_np).mean()
        random_baseline = 1.0 / self.trainer.n_actions

        print(f"\n📊 Inverse Model Performance:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   Random baseline: {random_baseline*100:.2f}%")
        print(f"   Improvement: {(accuracy - random_baseline)*100:.2f}%")

        if accuracy <= random_baseline * 1.1:
            print(f"   ⚠️  WARNING: Model is basically random guessing!")

        # Confusion matrix
        confusion = np.zeros((self.trainer.n_actions, self.trainer.n_actions))
        for true, pred in zip(actions_np, predicted_np):
            confusion[true, pred] += 1

        # Normalize by row (true action)
        confusion = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)

        # Plot confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion matrix
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.action_names, yticklabels=self.action_names,
                   ax=ax1, cbar_kws={'label': 'Probability'})
        ax1.set_title('Inverse Model Confusion Matrix\n(Row=True Action, Col=Predicted)')
        ax1.set_xlabel('Predicted Action')
        ax1.set_ylabel('True Action')

        # Per-action accuracy
        per_action_acc = confusion.diagonal()
        bars = ax2.bar(range(len(self.action_names)), per_action_acc)
        ax2.axhline(random_baseline, color='r', linestyle='--', label='Random baseline')
        ax2.set_xticks(range(len(self.action_names)))
        ax2.set_xticklabels(self.action_names, rotation=45)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Per-Action Accuracy')
        ax2.legend()
        ax2.set_ylim([0, 1])

        # Color bars by performance
        for bar, acc in zip(bars, per_action_acc):
            if acc > random_baseline * 1.5:
                bar.set_color('green')
            elif acc < random_baseline * 1.1:
                bar.set_color('red')
            else:
                bar.set_color('orange')

        plt.tight_layout()
        plt.savefig('diagnostic_02_inverse_model.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: diagnostic_02_inverse_model.png")
        plt.show()

        self.model.train()

        return {
            'accuracy': accuracy,
            'confusion_matrix': confusion,
            'per_action_accuracy': per_action_acc
        }

    # ========================================================================
    # 3. FORWARD MODEL DIAGNOSTIC
    # ========================================================================

    def diagnose_forward_model(self, n_samples=4):
        """
        Check if forward model predicts motion correctly.

        Shows:
        - Predicted vs actual next motion
        - Prediction error visualization
        """
        print("\n" + "="*70)
        print("3️⃣  FORWARD MODEL DIAGNOSTIC (PREDICTION)")
        print("="*70)

        if len(self.trainer.buffer) < n_samples:
            n_samples = len(self.trainer.buffer)

        self.model.eval()

        frames_t, actions, frames_t1 = self.trainer.buffer.sample(n_samples)
        frames_t = frames_t.to(self.device)
        frames_t1 = frames_t1.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            # Extract motion
            motion_t = self.model.extract_motion(frames_t, frames_t)
            motion_t1_actual = self.model.extract_motion(frames_t, frames_t1)

            # Predict next motion
            motion_t1_pred = self.model.forward_model(motion_t, actions)

        # Calculate errors
        errors = F.mse_loss(motion_t1_pred, motion_t1_actual, reduction='none').mean(dim=1)

        # Plot
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Frame pair
            ax0 = axes[i, 0]
            combined = torch.cat([frames_t[i], frames_t1[i]], dim=2)  # Side by side
            ax0.imshow(combined.cpu().permute(1, 2, 0).numpy())
            ax0.set_title(f'Frames t→t+1\nAction: {self.action_names[actions[i]]}')
            ax0.axis('off')

            # Actual motion
            ax1 = axes[i, 1]
            motion_actual_2d = motion_t1_actual[i].cpu().numpy().reshape(16, 8)
            im1 = ax1.imshow(motion_actual_2d, cmap='viridis')
            ax1.set_title('Actual Motion')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            # Predicted motion
            ax2 = axes[i, 2]
            motion_pred_2d = motion_t1_pred[i].cpu().numpy().reshape(16, 8)
            im2 = ax2.imshow(motion_pred_2d, cmap='viridis')
            ax2.set_title('Predicted Motion')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            # Error
            ax3 = axes[i, 3]
            error_2d = (motion_t1_actual[i] - motion_t1_pred[i]).cpu().abs().numpy().reshape(16, 8)
            im3 = ax3.imshow(error_2d, cmap='hot')
            ax3.set_title(f'Prediction Error\nMSE: {errors[i].item():.6f}')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)

        plt.tight_layout()
        plt.savefig('diagnostic_03_forward_model.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: diagnostic_03_forward_model.png")
        plt.show()

        self.model.train()

        print(f"\n📊 Forward Model Performance:")
        print(f"   Average MSE: {errors.mean().item():.6f}")
        print(f"   MSE std: {errors.std().item():.6f}")

    # ========================================================================
    # 4. CONSISTENCY CHECK
    # ========================================================================

    def check_consistency(self, n_samples=100):
        """
        Check if forward and inverse models agree on physics.

        Loop:
        1. motion_t, motion_t+1 (observed)
        2. action_inferred = Inverse(motion_t, motion_t+1)
        3. motion_t+1_reconstructed = Forward(motion_t, action_inferred)
        4. Error = |motion_t+1_reconstructed - motion_t+1|

        If models agree: Error should be LOW
        """
        print("\n" + "="*70)
        print("4️⃣  CONSISTENCY CHECK (Do models agree on physics?)")
        print("="*70)

        if len(self.trainer.buffer) < n_samples:
            n_samples = len(self.trainer.buffer)

        self.model.eval()

        frames_t, actions_true, frames_t1 = self.trainer.buffer.sample(n_samples)
        frames_t = frames_t.to(self.device)
        frames_t1 = frames_t1.to(self.device)

        with torch.no_grad():
            # Extract motion
            motion_t = self.model.extract_motion(frames_t, frames_t)
            motion_t1_actual = self.model.extract_motion(frames_t, frames_t1)

            # Consistency loop
            # 1. Infer action from motion
            action_logits = self.model.inverse_model(motion_t, motion_t1_actual)
            actions_inferred = torch.argmax(action_logits, dim=-1)

            # 2. Predict motion using inferred action
            motion_t1_reconstructed = self.model.forward_model(motion_t, actions_inferred)

            # 3. Compare to actual
            consistency_error = F.mse_loss(motion_t1_reconstructed, motion_t1_actual, reduction='none').mean(dim=1)

            # Also try with TRUE actions
            motion_t1_with_true_action = self.model.forward_model(motion_t, actions_true.to(self.device))
            true_action_error = F.mse_loss(motion_t1_with_true_action, motion_t1_actual, reduction='none').mean(dim=1)

        print(f"\n📊 Consistency Results:")
        print(f"   Consistency error (inferred action): {consistency_error.mean().item():.6f}")
        print(f"   Forward error (true action):         {true_action_error.mean().item():.6f}")
        print(f"   Ratio: {consistency_error.mean().item() / (true_action_error.mean().item() + 1e-8):.2f}x")

        if consistency_error.mean().item() > true_action_error.mean().item() * 2:
            print(f"   ⚠️  Models DISAGREE on physics!")
            print(f"       Inverse model infers wrong actions")
        else:
            print(f"   ✅ Models somewhat agree on physics")

        # Plot histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(consistency_error.cpu().numpy(), bins=30, alpha=0.7, label='Consistency (inferred action)')
        ax.hist(true_action_error.cpu().numpy(), bins=30, alpha=0.7, label='Forward (true action)')
        ax.axvline(consistency_error.mean().item(), color='blue', linestyle='--')
        ax.axvline(true_action_error.mean().item(), color='orange', linestyle='--')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Count')
        ax.set_title('Consistency vs Forward Error Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig('diagnostic_04_consistency.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: diagnostic_04_consistency.png")
        plt.show()

        self.model.train()

    # ========================================================================
    # 5. POLICY ANALYSIS
    # ========================================================================

    def analyze_policy(self, n_steps=100):
        """
        Check if policy is different from random.

        Shows:
        - Action distribution (policy vs random)
        - Action entropy over time
        """
        print("\n" + "="*70)
        print("5️⃣  POLICY ANALYSIS (Is it different from random?)")
        print("="*70)

        # Collect actions from policy
        env = self.trainer.env
        frame, _ = env.reset()
        frame = self.preprocess_frame(frame)

        policy_actions = []
        random_actions = []

        for step in range(n_steps):
            # Policy action
            policy_action = self.trainer.policy.select_action(frame, exploration_rate=0.1)
            policy_actions.append(policy_action)

            # Random action
            random_action = np.random.randint(0, self.trainer.n_actions)
            random_actions.append(random_action)

            # Execute policy action
            next_frame_raw, _, terminated, truncated, _ = env.step(policy_action)
            frame = self.preprocess_frame(next_frame_raw)

            if terminated or truncated:
                frame, _ = env.reset()
                frame = self.preprocess_frame(frame)

        # Analyze distributions
        policy_dist = np.bincount(policy_actions, minlength=self.trainer.n_actions) / len(policy_actions)
        random_dist = np.bincount(random_actions, minlength=self.trainer.n_actions) / len(random_actions)

        # Entropy
        policy_entropy = -np.sum(policy_dist * np.log(policy_dist + 1e-8))
        random_entropy = -np.sum(random_dist * np.log(random_dist + 1e-8))
        max_entropy = np.log(self.trainer.n_actions)

        print(f"\n📊 Policy Statistics:")
        print(f"   Policy entropy: {policy_entropy:.3f} / {max_entropy:.3f}")
        print(f"   Random entropy: {random_entropy:.3f} / {max_entropy:.3f}")
        print(f"   Entropy difference: {abs(policy_entropy - random_entropy):.3f}")

        if abs(policy_entropy - random_entropy) < 0.1:
            print(f"   ⚠️  Policy looks very similar to random!")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Action distributions
        x = np.arange(len(self.action_names))
        width = 0.35
        ax1.bar(x - width/2, policy_dist, width, label='Policy', alpha=0.8)
        ax1.bar(x + width/2, random_dist, width, label='Random', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.action_names, rotation=45)
        ax1.set_ylabel('Frequency')
        ax1.set_title('Action Distribution: Policy vs Random')
        ax1.legend()
        ax1.axhline(1.0/self.trainer.n_actions, color='r', linestyle='--', alpha=0.5, label='Uniform')

        # Action sequence
        ax2.plot(policy_actions[:50], 'o-', label='Policy', alpha=0.7)
        ax2.plot(random_actions[:50], 's-', label='Random', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Action')
        ax2.set_title('Action Sequence (First 50 steps)')
        ax2.set_yticks(range(self.trainer.n_actions))
        ax2.set_yticklabels(self.action_names)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('diagnostic_05_policy.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: diagnostic_05_policy.png")
        plt.show()

    # ========================================================================
    # FULL DIAGNOSTIC SUITE
    # ========================================================================

    def run_all_diagnostics(self):
        """Run complete diagnostic suite."""
        print("\n" + "="*80)
        print("🔬 RUNNING FULL DIAGNOSTIC SUITE")
        print("="*80)
        print("\nThis will check:")
        print("  1. Motion extraction (is it meaningful?)")
        print("  2. Inverse model (does it learn causality?)")
        print("  3. Forward model (does it predict correctly?)")
        print("  4. Consistency (do models agree?)")
        print("  5. Policy (different from random?)")
        print("="*80 + "\n")

        # Run all diagnostics
        try:
            self.visualize_motion_extraction(n_samples=4)
        except Exception as e:
            print(f"❌ Motion extraction failed: {e}")

        try:
            self.diagnose_inverse_model(n_samples=100)
        except Exception as e:
            print(f"❌ Inverse model diagnostic failed: {e}")

        try:
            self.diagnose_forward_model(n_samples=4)
        except Exception as e:
            print(f"❌ Forward model diagnostic failed: {e}")

        try:
            self.check_consistency(n_samples=100)
        except Exception as e:
            print(f"❌ Consistency check failed: {e}")

        try:
            self.analyze_policy(n_steps=100)
        except Exception as e:
            print(f"❌ Policy analysis failed: {e}")

        print("\n" + "="*80)
        print("✅ DIAGNOSTIC SUITE COMPLETE")
        print("="*80)
        print("\nCheck the generated images:")
        print("  - diagnostic_01_motion_extraction.png")
        print("  - diagnostic_02_inverse_model.png")
        print("  - diagnostic_03_forward_model.png")
        print("  - diagnostic_04_consistency.png")
        print("  - diagnostic_05_policy.png")
        print("="*80 + "\n")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 MOTION INVERSE MODEL - DIAGNOSTICS")
    print("="*70)
    print("\nLet's see if this is real or 'high fantasy'!")
    print("\nUsage:")
    print("  diagnostics = MotionDiagnostics(trainer)")
    print("  diagnostics.run_all_diagnostics()")
    print("="*70 + "\n")
