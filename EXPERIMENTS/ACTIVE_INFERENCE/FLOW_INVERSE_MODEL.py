"""
FLOW-BASED TETRAHEDRAL INVERSE MODEL
=====================================

Testing the Harmonic Resonance Hypothesis:
- Neural networks learn by tuning to harmonic frequencies
- Flow (temporal velocity) is the fundamental primitive
- Actions create characteristic frequency signatures in flow fields
- Tetrahedral structure learns four fundamental harmonics

This is THE test:
- Can inverse model predict actions from flow fields?
- Does it generalize better than frame-based approaches?
- Can we visualize which actions create which flow frequencies?

Architecture:
- Forward model: (flow_t, action) → flow_t+1
- Inverse model: (flow_t, flow_t+1) → action
- Both use DualTetrahedralNetwork
- Curriculum learning on flow visibility

Philosophy:
"Vision operates on movement. Stills are what you get when you integrate flow."

Author: Philipp Remy Bartholomäus & Claude
Date: November 12, 2025
Status: 🔬 Experimental validation of harmonic resonance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from collections import deque
import random
import cv2

# Assume W, X, Y, Z cells already run
# This imports DualTetrahedralNetwork

φ = (1 + 5**0.5) / 2  # Golden ratio

print("🌊 Flow-Based Tetrahedral Inverse Model")
print("=" * 70)
print("Testing: Harmonic Resonance Hypothesis")
print("Primitive: Flow (velocity fields), not frames")
print("=" * 70)


# ============================================================================
# VALID ACTIONS PER GAME (Critical for action masking!)
# ============================================================================

VALID_ACTIONS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN
    'ALE/Breakout-v5': [0, 1, 3, 4],  # NOOP, FIRE, RIGHT, LEFT
    'ALE/SpaceInvaders-v5': [0, 1, 2, 3, 4, 5],  # NOOP, FIRE, UP, RIGHT, LEFT, DOWN
    'ALE/MsPacman-v5': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # All 9 actions
    # Add more games as needed
}

ACTION_NAMES = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'UP',
    3: 'RIGHT',
    4: 'LEFT',
    5: 'DOWN',
    6: 'UPRIGHT',
    7: 'UPLEFT',
    8: 'DOWNRIGHT',
    9: 'DOWNLEFT',
    10: 'UPFIRE',
    11: 'RIGHTFIRE',
    12: 'LEFTFIRE',
    13: 'DOWNFIRE',
    14: 'UPRIGHTFIRE',
    15: 'UPLEFTFIRE',
    16: 'DOWNRIGHTFIRE',
    17: 'DOWNLEFTFIRE'
}


# ============================================================================
# OPTICAL FLOW COMPUTATION
# ============================================================================

def compute_optical_flow(frame1, frame2, method='farneback'):
    """
    Compute optical flow between two frames.

    Returns velocity field: (2, H, W) where:
    - Channel 0: horizontal velocity (flow_x)
    - Channel 1: vertical velocity (flow_y)

    This is the FUNDAMENTAL PRIMITIVE - movement itself.
    """
    # Convert to grayscale numpy arrays
    if torch.is_tensor(frame1):
        f1 = (frame1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        f2 = (frame2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    else:
        f1 = (frame1 * 255).astype(np.uint8)
        f2 = (frame2 * 255).astype(np.uint8)

    # Convert to grayscale
    if len(f1.shape) == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = f1
        gray2 = f2

    if method == 'farneback':
        # Dense optical flow (Farneback method)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    elif method == 'simple':
        # Simple finite difference (faster, less accurate)
        diff = (gray2.astype(float) - gray1.astype(float)) / 255.0
        flow = np.stack([diff, diff], axis=-1)  # Approximation

    # Convert to torch tensor (2, H, W)
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()

    return flow_tensor


def flow_to_rgb(flow):
    """
    Visualize flow field as RGB image (HSV color wheel encoding).

    - Hue: direction of flow
    - Saturation: magnitude of flow
    - Value: always 1

    Standard visualization in computer vision.
    """
    flow_np = flow.cpu().numpy() if torch.is_tensor(flow) else flow

    # flow_np shape: (2, H, W)
    u = flow_np[0]  # horizontal
    v = flow_np[1]  # vertical

    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    # Create HSV image
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi) * 179  # Hue: angle
    hsv[..., 1] = np.clip(mag * 50, 0, 255)           # Saturation: magnitude
    hsv[..., 2] = 255                                  # Value: max

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb / 255.0


# ============================================================================
# FLOW-BASED FORWARD MODEL
# ============================================================================

class FlowForwardModel(nn.Module):
    """
    Predict next flow field from current flow + action.

    (flow_t, action) → flow_t+1

    Learns: How do actions transform velocity fields?
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        # Action encoder
        self.action_encoder = nn.Embedding(n_actions, 64)

        # Tetrahedral network operates on flow
        input_dim = 2 * img_size * img_size + 64  # Flow (2 channels) + action
        output_dim = 2 * img_size * img_size       # Next flow (2 channels)

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, flow, action):
        """
        Predict next flow field.

        Args:
            flow: (batch, 2, H, W) - current velocity field
            action: (batch,) - action indices

        Returns:
            next_flow: (batch, 2, H, W) - predicted next velocity field
        """
        batch_size = flow.size(0)

        # Flatten flow
        flow_flat = flow.reshape(batch_size, -1)

        # Encode action
        action_embed = self.action_encoder(action)

        # Concatenate
        x = torch.cat([flow_flat, action_embed], dim=-1)

        # Predict next flow
        next_flow_flat = self.dual_tetra(x)
        next_flow = next_flow_flat.reshape(batch_size, 2, self.img_size, self.img_size)

        return next_flow


# ============================================================================
# FLOW-BASED INVERSE MODEL (THE KEY TEST!)
# ============================================================================

class FlowInverseModel(nn.Module):
    """
    Predict action from flow field changes.

    (flow_t, flow_t+1) → action

    THIS IS THE HARMONIC RESONANCE TEST:
    - Can it learn action frequency signatures?
    - "Upward flow in paddle region" → resonates with "UP action"
    - Does it generalize because it learned the harmonic?
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        # Tetrahedral network sees flow change
        # Input: concatenate flow_t and flow_t+1
        input_dim = 2 * 2 * img_size * img_size  # Two flow fields (4 channels total)
        output_dim = n_actions                    # Action logits

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, flow_t, flow_t1):
        """
        Infer action from flow field transition.

        Args:
            flow_t: (batch, 2, H, W) - current flow
            flow_t1: (batch, 2, H, W) - next flow

        Returns:
            action_logits: (batch, n_actions)
        """
        batch_size = flow_t.size(0)

        # Flatten both flows
        flow_t_flat = flow_t.reshape(batch_size, -1)
        flow_t1_flat = flow_t1.reshape(batch_size, -1)

        # Concatenate
        x = torch.cat([flow_t_flat, flow_t1_flat], dim=-1)

        # Predict action
        action_logits = self.dual_tetra(x)

        return action_logits


# ============================================================================
# COUPLED MODEL
# ============================================================================

class CoupledFlowModel(nn.Module):
    """
    Combined forward + inverse model operating on flow.

    Tests harmonic resonance:
    - Do they learn consistent frequency representations?
    - Does consistency loss couple the harmonics?
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        self.forward_model = FlowForwardModel(img_size, latent_dim, n_actions)
        self.inverse_model = FlowInverseModel(img_size, latent_dim, n_actions)

    def compute_losses(self, flow_t, flow_t1, action, action_mask=None,
                      forward_weight=1.0,
                      inverse_weight=1.0,
                      consistency_weight=0.3):
        """
        Compute coupled losses with ACTION MASKING.

        Critical: Mask invalid actions so model doesn't get penalized
        for predicting "UP" when ground truth is "UPFIRE" (functionally identical in Pong).

        Returns metrics for tracking harmonic learning.
        """
        # Forward loss: Can we predict next flow?
        pred_flow_t1 = self.forward_model(flow_t, action)
        loss_forward = F.mse_loss(pred_flow_t1, flow_t1)

        # Inverse loss: Can we infer action from flow change?
        action_logits = self.inverse_model(flow_t, flow_t1)

        # ===  ACTION MASKING (CRITICAL!) ===
        # Mask invalid actions to -inf before softmax/loss
        if action_mask is not None:
            masked_logits = action_logits.clone()
            # Expand mask to batch size
            mask_expanded = action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
            # Set invalid actions to very negative (will be ~0 after softmax)
            masked_logits[mask_expanded == 0] = -1e9
        else:
            masked_logits = action_logits

        loss_inverse = F.cross_entropy(masked_logits, action)

        # Consistency loss: Do models agree?
        with torch.no_grad():
            inferred_action = masked_logits.argmax(dim=-1)
        pred_flow_consistent = self.forward_model(flow_t, inferred_action)
        loss_consistency = F.mse_loss(pred_flow_consistent, flow_t1)

        # Total
        total_loss = (forward_weight * loss_forward +
                     inverse_weight * loss_inverse +
                     consistency_weight * loss_consistency)

        # Metrics (also use masked logits for accuracy!)
        with torch.no_grad():
            accuracy = (masked_logits.argmax(dim=-1) == action).float().mean()
            flow_magnitude = torch.sqrt((flow_t1 ** 2).sum(dim=1)).mean()

        return {
            'total': total_loss,
            'forward': loss_forward.item(),
            'inverse': loss_inverse.item(),
            'consistency': loss_consistency.item(),
            'accuracy': accuracy.item(),
            'flow_magnitude': flow_magnitude.item()
        }


# ============================================================================
# EXPERIENCE BUFFER (Flow-based)
# ============================================================================

class FlowExperienceBuffer:
    """Store (flow_t, action, flow_t+1) transitions."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, flow_t, action, flow_t1):
        self.buffer.append((flow_t, action, flow_t1))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        flows_t = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        flows_t1 = torch.stack([b[2] for b in batch])

        return flows_t, actions, flows_t1

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# TRAINER
# ============================================================================

class FlowInverseTrainer:
    """
    Train flow-based inverse model.

    Tests: Harmonic Resonance Hypothesis
    """
    def __init__(self,
                 env_name='ALE/Pong-v5',
                 img_size=128,
                 latent_dim=128,
                 base_lr=0.0001,
                 buffer_capacity=10000,
                 batch_size=16,
                 device=None,
                 flow_method='farneback'):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🖥️  Device: {self.device}")

        self.img_size = img_size
        self.batch_size = batch_size
        self.flow_method = flow_method
        self.env_name = env_name

        # Environment
        import gymnasium as gym
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.n_actions = self.env.action_space.n
        print(f"🎮 Environment: {env_name}")
        print(f"   Actions: {self.n_actions}")

        # === ACTION MASKING (CRITICAL!) ===
        # Create mask for valid actions in this game
        if env_name in VALID_ACTIONS:
            valid_actions = VALID_ACTIONS[env_name]
            self.action_mask = torch.zeros(self.n_actions)
            self.action_mask[valid_actions] = 1.0
            print(f"   Valid actions: {valid_actions}")
            print(f"   Action names: {[ACTION_NAMES.get(a, str(a)) for a in valid_actions]}")
        else:
            # No masking if game not in dictionary
            self.action_mask = torch.ones(self.n_actions)
            print(f"   ⚠️  No action mask for {env_name} - using all {self.n_actions} actions")

        self.action_mask = self.action_mask.to(self.device)

        # Model
        print(f"\n🧠 Creating coupled flow model...")
        self.model = CoupledFlowModel(
            img_size=img_size,
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {total_params:,}")

        # φ-hierarchical optimizer
        print(f"\n⚡ Creating φ-hierarchical optimizer...")
        vertex_params = []
        edge_params = []
        face_params = []
        coupling_params = []

        # Forward model
        dual_fwd = self.model.forward_model.dual_tetra
        vertex_params.extend(dual_fwd.linear_net.embed.parameters())
        vertex_params.extend(dual_fwd.nonlinear_net.embed.parameters())
        edge_params.extend(dual_fwd.linear_net.edge_modules.parameters())
        edge_params.extend(dual_fwd.nonlinear_net.edge_modules.parameters())
        face_params.extend(dual_fwd.linear_net.face_modules.parameters())
        face_params.extend(dual_fwd.nonlinear_net.face_modules.parameters())
        coupling_params.extend(dual_fwd.linear_to_nonlinear.parameters())
        coupling_params.extend(dual_fwd.nonlinear_to_linear.parameters())

        # Inverse model
        dual_inv = self.model.inverse_model.dual_tetra
        vertex_params.extend(dual_inv.linear_net.embed.parameters())
        vertex_params.extend(dual_inv.nonlinear_net.embed.parameters())
        edge_params.extend(dual_inv.linear_net.edge_modules.parameters())
        edge_params.extend(dual_inv.nonlinear_net.edge_modules.parameters())
        face_params.extend(dual_inv.linear_net.face_modules.parameters())
        face_params.extend(dual_inv.nonlinear_net.face_modules.parameters())
        coupling_params.extend(dual_inv.linear_to_nonlinear.parameters())
        coupling_params.extend(dual_inv.nonlinear_to_linear.parameters())

        # Action embedding
        action_params = list(self.model.forward_model.action_encoder.parameters())

        self.optimizer = optim.Adam([
            {'params': action_params, 'lr': base_lr},
            {'params': vertex_params, 'lr': base_lr},
            {'params': edge_params, 'lr': base_lr / φ},
            {'params': face_params, 'lr': base_lr / (φ ** 2)},
            {'params': coupling_params, 'lr': base_lr / (φ ** 3)}
        ])

        print(f"   Vertices:  LR = {base_lr:.6f}")
        print(f"   Edges:     LR = {base_lr/φ:.6f}")
        print(f"   Faces:     LR = {base_lr/(φ**2):.6f}")
        print(f"   Coupling:  LR = {base_lr/(φ**3):.6f}")

        # Buffer
        self.buffer = FlowExperienceBuffer(capacity=buffer_capacity)

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'forward': [], 'inverse': [], 'consistency': [],
            'total': [], 'accuracy': [], 'flow_magnitude': []
        }

    def preprocess_frame(self, frame):
        """Resize frame to img_size."""
        from PIL import Image
        img = Image.fromarray(frame).resize((self.img_size, self.img_size), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return tensor

    def collect_experience(self, n_steps=100):
        """Collect flow-based transitions (ONLY VALID ACTIONS!)."""
        print(f"\n📦 Collecting {n_steps} flow transitions...")

        # Get valid actions for this game
        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))

        frame_raw, _ = self.env.reset()
        frame_prev = self.preprocess_frame(frame_raw)
        episodes_done = 0

        for step in range(n_steps):
            # Random action from VALID set only!
            action = random.choice(valid_actions)

            # Step environment
            frame_raw, reward, terminated, truncated, info = self.env.step(action)
            frame_curr = self.preprocess_frame(frame_raw)

            # Compute flows
            flow_prev = compute_optical_flow(frame_prev, frame_curr, method=self.flow_method)

            # Next frame for next flow (also from valid actions!)
            action_next = random.choice(valid_actions)
            frame_raw_next, _, term_next, trunc_next, _ = self.env.step(action_next)
            frame_next = self.preprocess_frame(frame_raw_next)
            flow_curr = compute_optical_flow(frame_curr, frame_next, method=self.flow_method)

            # Store: (flow_prev, action, flow_curr)
            self.buffer.add(flow_prev, action, flow_curr)

            # Update
            frame_prev = frame_next

            if terminated or truncated or term_next or trunc_next:
                frame_raw, _ = self.env.reset()
                frame_prev = self.preprocess_frame(frame_raw)
                episodes_done += 1

        print(f"   ✓ Collected {len(self.buffer)} flow transitions")
        print(f"   ✓ Episodes: {episodes_done}")
        return episodes_done

    def train_step(self):
        """Train on batch of flow transitions WITH ACTION MASKING."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample
        flows_t, actions, flows_t1 = self.buffer.sample(self.batch_size)
        flows_t = flows_t.to(self.device)
        actions = actions.to(self.device)
        flows_t1 = flows_t1.to(self.device)

        # Compute losses WITH ACTION MASK!
        losses = self.model.compute_losses(
            flows_t, flows_t1, actions,
            action_mask=self.action_mask
        )

        # Backprop
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()

        self.step_count += 1

        return losses

    def train_loop(self, n_episodes=10, steps_per_episode=50, viz_every=2):
        """Main training loop WITH LIVE VISUALIZATION."""
        print("\n" + "="*70)
        print("🌊 FLOW-BASED INVERSE MODEL TRAINING")
        print("="*70)
        print(f"Testing: Harmonic Resonance Hypothesis")
        print(f"Episodes: {n_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Visualization every {viz_every} episodes")
        print("="*70 + "\n")

        for episode in range(n_episodes):
            # Collect
            episodes_done = self.collect_experience(n_steps=100)
            self.episode_count += episodes_done

            # Train
            print(f"\n🎓 Training for {steps_per_episode} steps...")
            ep_metrics = {'forward': [], 'inverse': [], 'accuracy': []}

            for _ in range(steps_per_episode):
                metrics = self.train_step()
                if metrics:
                    ep_metrics['forward'].append(metrics['forward'])
                    ep_metrics['inverse'].append(metrics['inverse'])
                    ep_metrics['accuracy'].append(metrics['accuracy'])

                    for key in self.history:
                        if key in metrics:
                            self.history[key].append(metrics[key])

            # Log
            if ep_metrics['forward']:
                avg_fwd = np.mean(ep_metrics['forward'])
                avg_inv = np.mean(ep_metrics['inverse'])
                avg_acc = np.mean(ep_metrics['accuracy'])

                print(f"\n📊 Episode {episode+1}/{n_episodes}")
                print(f"   Steps: {self.step_count}")
                print(f"   Forward loss: {avg_fwd:.6f}")
                print(f"   Inverse loss: {avg_inv:.6f}")
                print(f"   Accuracy: {avg_acc*100:.2f}% (random={100/self.n_actions:.2f}%)")
                print(f"   Buffer: {len(self.buffer)}")

            # === LIVE VISUALIZATION ===
            if (episode + 1) % viz_every == 0 and len(self.buffer) >= 4:
                print(f"\n🎨 Generating visualization (episode {episode+1})...")
                self._save_live_viz(episode + 1)

        print("\n✅ Training complete!\n")

    def _save_live_viz(self, episode_num):
        """Save live visualization during training."""
        self.model.eval()

        # Sample 4 transitions
        flows_t, actions, flows_t1 = self.buffer.sample(4)
        flows_t_dev = flows_t.to(self.device)
        flows_t1_dev = flows_t1.to(self.device)
        actions_dev = actions.to(self.device)

        with torch.no_grad():
            action_logits = self.model.inverse_model(flows_t_dev, flows_t1_dev)
            # Apply action mask
            masked_logits = action_logits.clone()
            mask_expanded = self.action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
            masked_logits[mask_expanded == 0] = -1e9
            pred_actions = masked_logits.argmax(dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)

        # Get valid actions for this game
        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))

        fig, axes = plt.subplots(4, 4, figsize=(14, 12))

        for i in range(4):
            # Flow t (as RGB)
            flow_t_rgb = flow_to_rgb(flows_t[i])
            axes[i, 0].imshow(flow_t_rgb)
            axes[i, 0].set_title(f'Flow t\nAction: {ACTION_NAMES.get(actions[i].item(), actions[i].item())}', fontsize=9)
            axes[i, 0].axis('off')

            # Flow t+1 (as RGB)
            flow_t1_rgb = flow_to_rgb(flows_t1[i])
            axes[i, 1].imshow(flow_t1_rgb)
            axes[i, 1].set_title('Flow t+1', fontsize=9)
            axes[i, 1].axis('off')

            # Flow change magnitude
            flow_diff = flows_t1[i] - flows_t[i]
            flow_diff_mag = torch.sqrt((flow_diff ** 2).sum(dim=0)).numpy()
            im = axes[i, 2].imshow(flow_diff_mag, cmap='hot')
            axes[i, 2].set_title('Flow change', fontsize=9)
            axes[i, 2].axis('off')

            # Action prediction (only show valid actions!)
            true_action = actions[i].item()
            pred_action = pred_actions[i].item()

            # Only show valid actions in bar chart
            valid_probs = probs[i, valid_actions].cpu().numpy()
            valid_names = [ACTION_NAMES.get(a, str(a)) for a in valid_actions]

            bars = axes[i, 3].bar(range(len(valid_actions)), valid_probs)

            # Color code bars
            if true_action in valid_actions:
                true_idx = valid_actions.index(true_action)
                bars[true_idx].set_color('green')
            if pred_action in valid_actions and pred_action != true_action:
                pred_idx = valid_actions.index(pred_action)
                bars[pred_idx].set_color('red')

            axes[i, 3].set_xticks(range(len(valid_actions)))
            axes[i, 3].set_xticklabels(valid_names, rotation=45, ha='right', fontsize=7)
            axes[i, 3].set_ylim([0, 1])
            axes[i, 3].axhline(1/len(valid_actions), color='gray', linestyle='--', alpha=0.5)

            correct = "✓" if pred_action == true_action else "✗"
            axes[i, 3].set_title(
                f'{correct} True: {ACTION_NAMES.get(true_action, true_action)}\nPred: {ACTION_NAMES.get(pred_action, pred_action)}',
                fontsize=8,
                color='green' if pred_action == true_action else 'red'
            )

        plt.suptitle(f'Episode {episode_num} | Step {self.step_count} | {self.env_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        filename = f'flow_live_ep{episode_num:03d}.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"   ✅ Saved: {filename}")
        plt.close()

        self.model.train()

    def visualize_flow_predictions(self, n_samples=4):
        """
        Visualize what the model learned.

        Shows:
        - Flow field (colored velocity arrows)
        - Predicted action
        - True action
        - Frequency signature (which vertex activates?)
        """
        if len(self.buffer) < n_samples:
            print("Not enough samples")
            return

        self.model.eval()

        flows_t, actions, flows_t1 = self.buffer.sample(n_samples)
        flows_t = flows_t.to(self.device)
        flows_t1 = flows_t1.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            action_logits = self.model.inverse_model(flows_t, flows_t1)
            pred_actions = action_logits.argmax(dim=-1)
            probs = torch.softmax(action_logits, dim=-1)

        # Atari action names (Pong subset)
        action_names = ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE']

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Flow t (as RGB)
            flow_t_rgb = flow_to_rgb(flows_t[i].cpu())
            axes[i, 0].imshow(flow_t_rgb)
            axes[i, 0].set_title('Flow t (velocity field)', fontsize=10)
            axes[i, 0].axis('off')

            # Flow t+1 (as RGB)
            flow_t1_rgb = flow_to_rgb(flows_t1[i].cpu())
            axes[i, 1].imshow(flow_t1_rgb)
            axes[i, 1].set_title('Flow t+1', fontsize=10)
            axes[i, 1].axis('off')

            # Flow change magnitude
            flow_diff = flows_t1[i] - flows_t[i]
            flow_diff_mag = torch.sqrt((flow_diff ** 2).sum(dim=0)).cpu().numpy()
            im = axes[i, 2].imshow(flow_diff_mag, cmap='hot')
            axes[i, 2].set_title('Flow change magnitude', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

            # Action prediction
            true_action = actions[i].item()
            pred_action = pred_actions[i].item()

            action_probs = probs[i].cpu().numpy()[:6]  # First 6 actions
            bars = axes[i, 3].bar(range(6), action_probs)

            # Color code
            bars[true_action].set_color('green')
            if pred_action != true_action and pred_action < 6:
                bars[pred_action].set_color('red')

            axes[i, 3].set_xticks(range(6))
            axes[i, 3].set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
            axes[i, 3].set_ylim([0, 1])
            axes[i, 3].axhline(1/self.n_actions, color='gray', linestyle='--', alpha=0.5, label='Random')

            correct = "✓" if pred_action == true_action else "✗"
            axes[i, 3].set_title(
                f'{correct} True: {action_names[true_action] if true_action < 6 else str(true_action)}\n'
                f'Pred: {action_names[pred_action] if pred_action < 6 else str(pred_action)}',
                fontsize=10,
                color='green' if pred_action == true_action else 'red'
            )

        plt.tight_layout()
        plt.savefig('flow_inverse_predictions.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: flow_inverse_predictions.png")
        plt.show()

        self.model.train()

    def plot_training(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Forward loss
        axes[0, 0].plot(self.history['forward'])
        axes[0, 0].set_title('Forward Loss (Flow Prediction)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(alpha=0.3)

        # Inverse loss
        axes[0, 1].plot(self.history['inverse'])
        axes[0, 1].axhline(np.log(self.n_actions), color='r', linestyle='--', label='Random')
        axes[0, 1].set_title('Inverse Loss (Action Prediction)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cross Entropy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Accuracy
        axes[1, 0].plot(np.array(self.history['accuracy']) * 100)
        axes[1, 0].axhline(100/self.n_actions, color='r', linestyle='--', label='Random')
        axes[1, 0].set_title('Inverse Model Accuracy')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Consistency
        axes[1, 1].plot(self.history['consistency'])
        axes[1, 1].set_title('Consistency Loss (Forward-Inverse Agreement)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('flow_training_curves.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: flow_training_curves.png")
        plt.show()


# ============================================================================
# QUICK START
# ============================================================================

print("\n" + "="*70)
print("🌊 FLOW-BASED TETRAHEDRAL INVERSE MODEL - READY")
print("="*70)
print("\nUsage:")
print("  trainer = FlowInverseTrainer(env_name='ALE/Pong-v5')")
print("  trainer.train_loop(n_episodes=10)")
print("  trainer.visualize_flow_predictions()")
print("  trainer.plot_training()")
print("\nTesting: Harmonic Resonance Hypothesis")
print("  - Do actions create characteristic flow frequencies?")
print("  - Does tetrahedral structure learn harmonics?")
print("  - Does it generalize beyond training?")
print("\n🔬 Let's find out!")
print("="*70 + "\n")
