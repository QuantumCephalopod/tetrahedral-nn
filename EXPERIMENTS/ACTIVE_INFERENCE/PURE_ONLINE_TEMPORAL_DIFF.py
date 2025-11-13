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


def preprocess_frame(frame):
    """
    Convert to grayscale, normalize.
    Simple. No fancy preprocessing.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    # Resize to square (pad, don't stretch!)
    h, w = gray.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=gray.dtype)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = gray

    # Normalize
    normalized = padded.astype(np.float32) / 255.0

    return torch.from_numpy(normalized)


def compute_temporal_difference(frame_t, frame_t1):
    """
    Compute temporal difference: frame_t+1 - frame_t

    This is what photoreceptors see.
    Positive = brightening, Negative = darkening

    Returns: (1, H, W) tensor
    """
    diff = frame_t1 - frame_t
    return diff.unsqueeze(0)  # Add channel dimension


class TemporalDifferenceModel(nn.Module):
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

        # Build model
        print(f"🔷 Building model...")
        self.model = TemporalDifferenceModel(
            img_size=img_size,
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)

        print(f"✅ Model ready ({sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params)")

        # History
        self.history = {'loss': [], 'accuracy': []}
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
        frame_prev = preprocess_frame(frame_raw)

        # Live visualization
        if show_gameplay:
            import matplotlib.pyplot as plt
            from IPython import display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.ion()

        # Bootstrap: get first temporal difference
        action = random.choice(self.valid_actions)
        frame_raw, _, terminated, truncated, _ = self.env.step(action)
        frame_curr = preprocess_frame(frame_raw)
        temporal_diff_prev = compute_temporal_difference(frame_prev, frame_curr)

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
            frame_next = preprocess_frame(frame_raw)
            temporal_diff_curr = compute_temporal_difference(frame_curr, frame_next)

            # ===== VISUALIZE =====
            if show_gameplay and step % 2 == 0:
                axes[0].clear()
                axes[0].imshow(frame_raw)
                axes[0].set_title(f'Step {step} | {ACTION_NAMES.get(action, action)}')
                axes[0].axis('off')

                axes[1].clear()
                diff_viz = temporal_diff_curr.squeeze().numpy()
                axes[1].imshow(diff_viz, cmap='RdBu', vmin=-1, vmax=1)
                axes[1].set_title('Temporal Difference')
                axes[1].axis('off')

                if len(self.history['loss']) > 10:
                    axes[2].clear()
                    recent_loss = self.history['loss'][-100:]
                    axes[2].plot(recent_loss, color='blue')
                    axes[2].set_title(f'Loss: {recent_loss[-1]:.4f}')
                    axes[2].set_xlabel('Recent Steps')
                    axes[2].grid(alpha=0.3)

                plt.tight_layout()
                display.clear_output(wait=True)
                display.display(fig)

            # ===== LEARN IMMEDIATELY =====
            diff_t = temporal_diff_prev.unsqueeze(0).to(self.device)
            actions = torch.tensor([action], dtype=torch.long).to(self.device)
            diff_t1 = temporal_diff_curr.unsqueeze(0).to(self.device)

            # Forward prediction
            pred_diff = self.model(diff_t, actions)
            loss = F.mse_loss(pred_diff, diff_t1)

            # Update
            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip
                )

            self.optimizer.step()

            # Track
            self.history['loss'].append(loss.item())

            if step % log_every == 0:
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Episodes: {episodes_done}")

            # ===== CONTINUE =====
            temporal_diff_prev = temporal_diff_curr
            frame_prev = frame_curr
            frame_curr = frame_next

            step += 1
            self.step_count += 1

            # Reset if episode ended
            if terminated or truncated:
                frame_raw, _ = self.env.reset()
                frame_prev = preprocess_frame(frame_raw)

                # Bootstrap
                action = random.choice(self.valid_actions)
                frame_raw, _, _, _, _ = self.env.step(action)
                frame_curr = preprocess_frame(frame_raw)
                temporal_diff_prev = compute_temporal_difference(frame_prev, frame_curr)

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
