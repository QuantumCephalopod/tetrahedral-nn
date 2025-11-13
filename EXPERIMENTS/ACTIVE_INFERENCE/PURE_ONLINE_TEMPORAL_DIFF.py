"""
PURE ONLINE LEARNING - TEMPORAL DIFFERENCES
============================================

No optical flow preprocessing. Just raw temporal differences.

This is what photoreceptors actually see: dI/dt (change in intensity over time)

The tetrahedral network learns motion structure from these raw signals,
just like the visual system does.

Philosophy: Nature already figured this out. Give the network the real signal.

Author: Philipp Remy Bartholomäus & Claude
Date: November 12, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import random
import gymnasium as gym

# Assumes W, X, Y, Z cells define DualTetrahedralNetwork
# Import from Z cell
from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork

φ = (1 + 5**0.5) / 2  # Golden ratio

VALID_ACTIONS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN
    'ALE/Breakout-v5': [0, 1, 3, 4],  # NOOP, FIRE, RIGHT, LEFT
}

ACTION_NAMES = {
    0: 'NOOP', 1: 'FIRE', 2: 'UP', 3: 'RIGHT',
    4: 'LEFT', 5: 'DOWN', 6: 'UPRIGHT', 7: 'UPLEFT'
}


def preprocess_frame(frame, target_size=210):
    """
    Convert to grayscale, pad to square, upscale to target size.

    Args:
        frame: Atari frame (210×160×3 typically)
        target_size: Target resolution (e.g., 420 for 2× upscale)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    # Pad to square (don't stretch!)
    h, w = gray.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=gray.dtype)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = gray

    # Upscale to target size if needed (for sub-pixel precision!)
    if size != target_size:
        padded = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Normalize
    normalized = padded.astype(np.float32) / 255.0

    return torch.from_numpy(normalized)


def add_microsaccade(frame, step_count, amplitude=1.5):
    """
    Add biological microsaccade: infinity symbol (figure-8) motion.

    Purpose: Make static objects visible (they produce diff=0 otherwise)

    Instead of random jitter, trace a smooth infinity symbol that:
    - Covers all directions uniformly
    - Continuous smooth motion (like real eye movements)
    - ~1-2 Hz frequency when traced

    Args:
        frame: torch tensor (H, W)
        step_count: current training step (for trajectory phase)
        amplitude: motion amplitude in pixels (~1.5 for sub-pixel at 420×420)

    Returns:
        Shifted frame tracing infinity symbol
    """
    # Infinity symbol trajectory (Lissajous curve with 1:2 frequency ratio)
    t = step_count * 0.1  # Slow smooth motion
    dx = amplitude * np.sin(t)           # Horizontal: frequency f
    dy = amplitude * np.sin(2 * t) / 2   # Vertical: frequency 2f, half amplitude

    # This traces a figure-8 that covers all directions!
    dx = int(np.round(dx))
    dy = int(np.round(dy))

    if dx == 0 and dy == 0:
        return frame  # No shift needed

    # Shift (using numpy for cv2)
    frame_np = frame.numpy()
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(frame_np, M, (frame_np.shape[1], frame_np.shape[0]))

    return torch.from_numpy(shifted)


def compute_temporal_difference(frame_t, frame_t1, add_saccade=True, step_count=0):
    """
    Compute temporal difference: frame_t+1 - frame_t

    This is what photoreceptors see.
    Positive = brightening, Negative = darkening

    Args:
        add_saccade: If True, apply infinity-symbol microsaccade
        step_count: Current step (for saccade trajectory)

    Returns: (1, H, W) tensor
    """
    # Infinity symbol microsaccade (reveals static objects)
    if add_saccade:
        frame_t1 = add_microsaccade(frame_t1, step_count)

    diff = frame_t1 - frame_t
    return diff.unsqueeze(0)  # Add channel dimension


class ForwardModel(nn.Module):
    """
    Forward model: predict temporal difference from current state + action

    (temporal_diff_t, action) → temporal_diff_t+1
    """
    def __init__(self, img_size=210, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        # Action embedding
        self.action_encoder = nn.Embedding(n_actions, 64)

        # Tetrahedral network
        input_dim = img_size * img_size + 64  # Temporal diff (1 channel) + action
        output_dim = img_size * img_size      # Next temporal diff

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, temporal_diff, action):
        """
        Predict next temporal difference.

        Args:
            temporal_diff: (batch, 1, H, W)
            action: (batch,)

        Returns:
            next_temporal_diff: (batch, 1, H, W)
        """
        batch_size = temporal_diff.size(0)

        # Flatten temporal diff
        diff_flat = temporal_diff.reshape(batch_size, -1)

        # Encode action
        action_embed = self.action_encoder(action)

        # Concatenate
        x = torch.cat([diff_flat, action_embed], dim=-1)

        # Predict next temporal diff
        next_diff_flat = self.dual_tetra(x)
        next_diff = next_diff_flat.reshape(batch_size, 1, self.img_size, self.img_size)

        return next_diff


class InverseModel(nn.Module):
    """
    Inverse model: infer action from temporal difference changes

    (temporal_diff_t, temporal_diff_t+1) → action

    This forces the forward model to encode meaningful change!
    Can't predict action from "nothing changed" → must represent motion.
    """
    def __init__(self, img_size=210, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        # Tetrahedral network sees TWO temporal diffs
        input_dim = 2 * img_size * img_size  # Two temporal diff fields
        output_dim = n_actions                # Action logits

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, temporal_diff_t, temporal_diff_t1):
        """
        Infer action from temporal difference transition.

        Args:
            temporal_diff_t: (batch, 1, H, W)
            temporal_diff_t1: (batch, 1, H, W)

        Returns:
            action_logits: (batch, n_actions)
        """
        batch_size = temporal_diff_t.size(0)

        # Flatten both diffs
        diff_t_flat = temporal_diff_t.reshape(batch_size, -1)
        diff_t1_flat = temporal_diff_t1.reshape(batch_size, -1)

        # Concatenate
        x = torch.cat([diff_t_flat, diff_t1_flat], dim=-1)

        # Predict action
        action_logits = self.dual_tetra(x)

        return action_logits


class CoupledModel(nn.Module):
    """
    Coupled forward + inverse models.

    The consistency loss between them prevents mode collapse:
    - Forward can't just predict "nothing changes"
    - Because inverse needs to infer actions from changes
    - They constrain each other!
    """
    def __init__(self, img_size=210, latent_dim=128, n_actions=18):
        super().__init__()
        self.forward_model = ForwardModel(img_size, latent_dim, n_actions)
        self.inverse_model = InverseModel(img_size, latent_dim, n_actions)

    def forward(self, temporal_diff, action):
        """Convenience: just call forward model"""
        return self.forward_model(temporal_diff, action)

    def compute_losses(self, diff_t, diff_t1, action, action_mask=None):
        """
        Compute coupled losses:
        1. Forward: predict next temporal diff (weighted by signal magnitude)
        2. Inverse: infer action from temporal diffs
        3. Consistency: do they agree?

        This coupling prevents mode collapse!
        """
        # Forward loss: predict next temporal diff
        pred_diff_t1 = self.forward_model(diff_t, action)

        # Weight prediction error by signal magnitude
        # Neurons fire for change, not for nothing!
        # High motion → high weight, static background → low weight
        signal_magnitude = torch.abs(diff_t1)
        weighted_error = signal_magnitude * (pred_diff_t1 - diff_t1) ** 2
        loss_forward = weighted_error.mean()

        # Inverse loss: infer action from temporal diffs
        action_logits = self.inverse_model(diff_t, diff_t1)

        # Apply action mask (only valid actions)
        if action_mask is not None:
            masked_logits = action_logits.clone()
            mask_expanded = action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
            masked_logits[mask_expanded == 0] = -1e9
        else:
            masked_logits = action_logits

        loss_inverse = F.cross_entropy(masked_logits, action)

        # Consistency loss: if inverse predicts action, forward should match reality
        with torch.no_grad():
            inferred_action = masked_logits.argmax(dim=-1)

        pred_diff_consistent = self.forward_model(diff_t, inferred_action)

        # Also weight consistency by signal magnitude
        weighted_error_consistency = signal_magnitude * (pred_diff_consistent - diff_t1) ** 2
        loss_consistency = weighted_error_consistency.mean()

        # Accuracy
        with torch.no_grad():
            accuracy = (inferred_action == action).float().mean().item()

        return {
            'forward': loss_forward,
            'inverse': loss_inverse,
            'consistency': loss_consistency,
            'total': loss_forward + loss_inverse + 0.3 * loss_consistency,
            'accuracy': accuracy
        }


class PureOnlineTrainer:
    """
    Pure online learning with temporal differences.

    No buffer. No batching. No optical flow preprocessing.
    Just: Act → Learn → Act → Learn
    """
    def __init__(self,
                 env_name='ALE/Pong-v5',
                 img_size=210,
                 latent_dim=128,
                 base_lr=0.0001,
                 device=None,
                 frameskip=3,
                 gradient_clip=1.0):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🌊 PURE ONLINE - TEMPORAL DIFFERENCES")
        print(f"🖥️  Device: {self.device}")
        print(f"🎯 Resolution: {img_size}×{img_size}")

        self.img_size = img_size
        self.env_name = env_name
        self.frameskip = frameskip
        self.gradient_clip = gradient_clip

        print(f"⏰ Frameskip: {frameskip} ({60.0/frameskip:.1f} Hz)")

        # Environment
        self.env = gym.make(env_name, frameskip=frameskip, render_mode=None)

        # Valid actions
        self.valid_actions = VALID_ACTIONS.get(env_name, list(range(self.env.action_space.n)))
        self.n_actions = self.env.action_space.n
        print(f"🎮 Valid actions: {[ACTION_NAMES.get(a, a) for a in self.valid_actions]}")

        # Action mask
        self.action_mask = torch.zeros(self.n_actions)
        self.action_mask[self.valid_actions] = 1.0
        self.action_mask = self.action_mask.to(self.device)

        # Build coupled model (forward + inverse)
        print(f"🔷 Building coupled model (input: {img_size}×{img_size})...")
        self.model = CoupledModel(
            img_size=img_size,  # This MUST match the actual frame size!
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)

        print(f"✅ Coupled model ready ({sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params)")
        print(f"   Forward + Inverse + Consistency (prevents mode collapse!)")

        # History
        self.history = {
            'loss': [],
            'forward': [],
            'inverse': [],
            'consistency': [],
            'accuracy': []
        }
        self.step_count = 0

        print(f"🌊 Initialized - NO BUFFER, NO FLOW PREPROCESSING")
        print(f"   Signal: Raw temporal differences (what photoreceptors see)")
        print(f"   Learning: φ-memory fields provide temporal integration")

    def select_action_active_inference(self, temporal_diff, temperature=1.0, beta=0.05):
        """
        Active inference: predict temporal difference for each action,
        choose action that minimizes free energy.
        """
        temporal_diff_tensor = temporal_diff.unsqueeze(0).to(self.device)

        free_energies = []
        for action in self.valid_actions:
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)

            with torch.no_grad():
                pred_diff = self.model(temporal_diff_tensor, action_tensor)

                # Uncertainty
                uncertainty = pred_diff.var().item()

                # Entropy
                entropy = pred_diff.std().item()

                # Expected Free Energy
                efe = uncertainty - beta * entropy

                free_energies.append(efe if np.isfinite(efe) else 1e6)

        # Softmax selection
        free_energies = np.array(free_energies)
        probs = np.exp(-free_energies / temperature)
        probs = probs / probs.sum()

        action_idx = np.random.choice(len(self.valid_actions), p=probs)
        return self.valid_actions[action_idx]

    def train(self, n_steps=1000, policy_temperature=1.0, beta=0.05,
              show_gameplay=True, log_every=50):
        """
        Pure online training loop.

        Act → Learn IMMEDIATELY → Act → Learn → ...
        """
        print("\n" + "="*70)
        print("🌀 STARTING PURE ONLINE LEARNING")
        print("="*70)

        # Initialize
        frame_raw, info = self.env.reset()
        frame_prev = preprocess_frame(frame_raw, target_size=self.img_size)

        # Live visualization
        if show_gameplay:
            import matplotlib.pyplot as plt
            from IPython import display
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            plt.ion()

        # Bootstrap: get first temporal difference
        action = random.choice(self.valid_actions)
        frame_raw, _, terminated, truncated, _ = self.env.step(action)
        frame_curr = preprocess_frame(frame_raw, target_size=self.img_size)
        temporal_diff_prev = compute_temporal_difference(frame_prev, frame_curr, step_count=0)

        print(f"🌊 Starting...\n")

        step = 0
        episodes_done = 0

        while step < n_steps:
            # ===== SELECT ACTION =====
            if step > 0:
                action = self.select_action_active_inference(
                    temporal_diff_prev,
                    temperature=policy_temperature,
                    beta=beta
                )
            else:
                action = random.choice(self.valid_actions)

            # ===== EXECUTE =====
            frame_raw, _, terminated, truncated, _ = self.env.step(action)
            frame_next = preprocess_frame(frame_raw, target_size=self.img_size)
            temporal_diff_curr = compute_temporal_difference(frame_curr, frame_next, step_count=step)

            # ===== VISUALIZE =====
            if show_gameplay and step % 2 == 0:
                # Row 1: Game, Input, Ground Truth, Forward Prediction
                axes[0, 0].clear()
                axes[0, 0].imshow(frame_raw)
                axes[0, 0].set_title(f'Step {step} | Action: {ACTION_NAMES.get(action, action)}')
                axes[0, 0].axis('off')

                axes[0, 1].clear()
                input_viz = temporal_diff_prev.squeeze().numpy()
                axes[0, 1].imshow(input_viz, cmap='RdBu', vmin=-1, vmax=1)
                axes[0, 1].set_title('Input: Temporal Diff t→t+1')
                axes[0, 1].axis('off')

                axes[0, 2].clear()
                truth_viz = temporal_diff_curr.squeeze().numpy()
                axes[0, 2].imshow(truth_viz, cmap='RdBu', vmin=-1, vmax=1)
                axes[0, 2].set_title('Ground Truth: Diff t+1→t+2')
                axes[0, 2].axis('off')

                # Row 2: Forward Prediction, Inverse Prediction, Errors, Loss
                if step > 0:
                    # Get predictions
                    with torch.no_grad():
                        diff_t = temporal_diff_prev.unsqueeze(0).to(self.device)
                        diff_t1 = temporal_diff_curr.unsqueeze(0).to(self.device)
                        actions_tensor = torch.tensor([action], dtype=torch.long).to(self.device)

                        # Forward prediction
                        pred_diff = self.model.forward_model(diff_t, actions_tensor)
                        pred_viz = pred_diff.squeeze().cpu().numpy()

                        # Inverse prediction
                        action_logits = self.model.inverse_model(diff_t, diff_t1)
                        if self.action_mask is not None:
                            masked_logits = action_logits.clone()
                            mask_expanded = self.action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
                            masked_logits[mask_expanded == 0] = -1e9
                        else:
                            masked_logits = action_logits
                        inferred_action = masked_logits.argmax(dim=-1).item()
                        action_probs = torch.softmax(masked_logits, dim=-1).squeeze().cpu().numpy()

                    # Forward prediction
                    axes[0, 3].clear()
                    axes[0, 3].imshow(pred_viz, cmap='RdBu', vmin=-1, vmax=1)
                    axes[0, 3].set_title('Forward Prediction: Diff t+1→t+2')
                    axes[0, 3].axis('off')

                    # Forward error
                    error_viz = np.abs(truth_viz - pred_viz)
                    axes[1, 0].clear()
                    axes[1, 0].imshow(error_viz, cmap='hot', vmin=0, vmax=0.5)
                    axes[1, 0].set_title(f'Forward Error (MAE: {error_viz.mean():.4f})')
                    axes[1, 0].axis('off')

                    # Inverse prediction (action probabilities)
                    axes[1, 1].clear()
                    valid_action_names = [ACTION_NAMES.get(a, str(a)) for a in self.valid_actions]
                    valid_probs = action_probs[self.valid_actions]
                    axes[1, 1].bar(range(len(self.valid_actions)), valid_probs)
                    axes[1, 1].set_xticks(range(len(self.valid_actions)))
                    axes[1, 1].set_xticklabels(valid_action_names, rotation=45)
                    axes[1, 1].set_ylim([0, 1])
                    axes[1, 1].set_title(f'Inverse: {ACTION_NAMES.get(inferred_action, inferred_action)}' +
                                        (f' ✓' if inferred_action == action else f' ✗'))
                    axes[1, 1].grid(alpha=0.3)

                    # Loss curves
                    if len(self.history['forward']) > 10:
                        axes[1, 2].clear()
                        recent_fwd = self.history['forward'][-100:]
                        recent_inv = self.history['inverse'][-100:]
                        axes[1, 2].plot(recent_fwd, label='Forward', color='blue', linewidth=2)
                        axes[1, 2].plot(recent_inv, label='Inverse', color='orange', linewidth=2)
                        axes[1, 2].set_title(f'Fwd: {recent_fwd[-1]:.5f} | Inv: {recent_inv[-1]:.5f}')
                        axes[1, 2].set_xlabel('Recent Steps')
                        axes[1, 2].legend()
                        axes[1, 2].grid(alpha=0.3)

                    # Accuracy curve
                    if len(self.history['accuracy']) > 10:
                        axes[1, 3].clear()
                        recent_acc = self.history['accuracy'][-100:]
                        axes[1, 3].plot(recent_acc, color='green', linewidth=2)
                        axes[1, 3].axhline(1/len(self.valid_actions), color='red',
                                          linestyle='--', label='Random', alpha=0.5)
                        axes[1, 3].set_ylim([0, 1])
                        axes[1, 3].set_title(f'Inverse Accuracy: {recent_acc[-1]*100:.1f}%')
                        axes[1, 3].set_xlabel('Recent Steps')
                        axes[1, 3].legend()
                        axes[1, 3].grid(alpha=0.3)

                plt.tight_layout()
                display.clear_output(wait=True)
                display.display(fig)

            # ===== LEARN IMMEDIATELY =====
            diff_t = temporal_diff_prev.unsqueeze(0).to(self.device)
            actions = torch.tensor([action], dtype=torch.long).to(self.device)
            diff_t1 = temporal_diff_curr.unsqueeze(0).to(self.device)

            # Compute coupled losses (forward + inverse + consistency)
            losses = self.model.compute_losses(diff_t, diff_t1, actions, action_mask=self.action_mask)

            # Update
            self.optimizer.zero_grad()
            losses['total'].backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip
                )

            self.optimizer.step()

            # Track
            self.history['loss'].append(losses['total'].item())
            self.history['forward'].append(losses['forward'].item())
            self.history['inverse'].append(losses['inverse'].item())
            self.history['consistency'].append(losses['consistency'].item())
            self.history['accuracy'].append(losses['accuracy'])

            if step % log_every == 0:
                print(f"Step {step:4d} | Fwd: {losses['forward'].item():.4f} | "
                      f"Inv: {losses['inverse'].item():.4f} | "
                      f"Acc: {losses['accuracy']*100:.1f}% | Eps: {episodes_done}")

            # ===== CONTINUE =====
            temporal_diff_prev = temporal_diff_curr
            frame_prev = frame_curr
            frame_curr = frame_next

            step += 1
            self.step_count += 1

            # Reset if episode ended
            if terminated or truncated:
                frame_raw, _ = self.env.reset()
                frame_prev = preprocess_frame(frame_raw, target_size=self.img_size)

                # Bootstrap
                action = random.choice(self.valid_actions)
                frame_raw, _, _, _, _ = self.env.step(action)
                frame_curr = preprocess_frame(frame_raw, target_size=self.img_size)
                temporal_diff_prev = compute_temporal_difference(frame_prev, frame_curr, step_count=step)

                episodes_done += 1

        if show_gameplay:
            plt.ioff()
            plt.close(fig)

        print("\n" + "="*70)
        print(f"✅ Training complete - {step} steps, {episodes_done} episodes")
        print("="*70)

        return self.history


print("""
TO USE:
=======

trainer = PureOnlineTrainer(
    env_name='ALE/Pong-v5',
    img_size=210,
    latent_dim=128,
    frameskip=3
)

history = trainer.train(n_steps=1000, show_gameplay=True)
""")
