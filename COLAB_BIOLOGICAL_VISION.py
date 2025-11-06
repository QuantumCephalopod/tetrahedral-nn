"""
🧠 BIOLOGICAL VISION - COLAB VERSION
=====================================

JUST PASTE THIS ENTIRE CELL INTO COLAB AND RUN IT!

This is ONE COMPLETE CELL - no external files needed.

What it does:
1. Creates a test video
2. Shows biological vision with visualization
3. Time is LINEAR, Space is NON-LINEAR (like real eyes!)
"""

# ============================================================================
# STEP 1: Install dependencies (if needed)
# ============================================================================

try:
    import torch
except:
    print("📦 Installing PyTorch...")
    !pip install -q torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
from IPython.display import clear_output, display, Image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import io

print("✓ All dependencies loaded!")

# ============================================================================
# STEP 2: Create Test Video
# ============================================================================

def create_test_video(output_path="test_video.mp4", duration_seconds=5):
    """Create a test video with moving shapes."""
    print("📹 Creating test video...")

    fps = 30
    width = 640
    height = 480
    total_frames = fps * duration_seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Bouncing ball
        circle_x = int(width/2 + 200 * np.sin(frame_idx * 0.1))
        circle_y = int(height/2 + 150 * np.cos(frame_idx * 0.15))
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 255), -1)

        # Sliding rectangle
        rect_x = int((frame_idx * 5) % width)
        rect_y = int(height/3)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 50, rect_y + 50), (255, 0, 255), -1)

        # Rotating line
        angle = frame_idx * 0.1
        center = (width//2, height//2)
        end_x = int(center[0] + 150 * np.cos(angle))
        end_y = int(center[1] + 150 * np.sin(angle))
        cv2.line(frame, center, (end_x, end_y), (255, 255, 0), 3)

        out.write(frame)

    out.release()
    print(f"✓ Test video created: {output_path}")
    return output_path


# ============================================================================
# STEP 3: Tetrahedral Neural Network (from your project)
# ============================================================================

class DualTetrahedralNetwork(nn.Module):
    """Simplified tetrahedral network for this demo."""

    def __init__(self, input_dim, output_dim, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


# ============================================================================
# STEP 4: Biological Attention System
# ============================================================================

class BiologicalAttention:
    """Non-linear spatial attention (like real eyes)."""

    def __init__(self, window_size=(128, 128), center_bias=0.7):
        self.window_size = window_size
        self.center_bias = center_bias
        self.frame_size = None
        self.current_position = None

    def set_frame_size(self, frame_size):
        """Set frame size and start at center."""
        self.frame_size = frame_size
        center_x = (frame_size[0] - self.window_size[0]) // 2
        center_y = (frame_size[1] - self.window_size[1]) // 2
        self.current_position = (center_x, center_y)

    def next_position(self):
        """Generate next attention position (saccade)."""
        frame_w, frame_h = self.frame_size
        win_w, win_h = self.window_size

        center_x = (frame_w - win_w) // 2
        center_y = (frame_h - win_h) // 2

        # Center-biased or exploratory?
        if np.random.random() < self.center_bias:
            # Look near center (fovea)
            x = int(np.random.normal(center_x, 100))
            y = int(np.random.normal(center_y, 100))
        else:
            # Exploratory saccade
            dx = np.random.normal(0, 100)
            dy = np.random.normal(0, 100)
            x = int(self.current_position[0] + dx)
            y = int(self.current_position[1] + dy)

        # Keep in bounds
        x = np.clip(x, 0, frame_w - win_w)
        y = np.clip(y, 0, frame_h - win_h)

        self.current_position = (x, y)
        return (x, y)

    def extract_window(self, frame, position):
        """Extract attention window from frame."""
        x, y = position
        win_w, win_h = self.window_size

        if len(frame.shape) == 2:
            return frame[y:y+win_h, x:x+win_w]
        else:
            return frame[y:y+win_h, x:x+win_w, :]


# ============================================================================
# STEP 5: Biological Vision Learner
# ============================================================================

class BiologicalVisionLearner:
    """Learns from biological vision stream."""

    def __init__(self, model, optimizer, window_size=3):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size

        self.frame_buffer = deque(maxlen=window_size + 1)
        self.loss_history = deque(maxlen=1000)
        self.current_loss = 0.0
        self.total_updates = 0

    def observe_frame(self, frame):
        """Observe a frame and learn."""
        self.frame_buffer.append(frame)

        # Need enough frames to learn
        if len(self.frame_buffer) < self.window_size + 1:
            return None

        # Prepare input (first 3 frames) and target (4th frame)
        input_frames = list(self.frame_buffer)[:self.window_size]
        target_frame = self.frame_buffer[self.window_size]

        # Flatten
        input_flat = np.concatenate([f.flatten() for f in input_frames])
        target_flat = target_frame.flatten()

        # To tensors
        input_tensor = torch.tensor(input_flat, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_flat, dtype=torch.float32).unsqueeze(0)

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(input_tensor)

        # Compute loss
        loss = nn.functional.mse_loss(prediction, target_tensor)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Track
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.current_loss = loss_value
        self.total_updates += 1

        return loss_value


# ============================================================================
# STEP 6: Visualization for Colab
# ============================================================================

def visualize_biological_vision_colab(video_path, max_frames=100, update_every=5):
    """
    Show biological vision with Colab-compatible visualization.
    """
    print("\n" + "=" * 70)
    print("👁️  BIOLOGICAL VISION SYSTEM")
    print("=" * 70)
    print("Time is LINEAR → Space is NON-LINEAR")
    print("=" * 70)
    print()

    # Create system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    attention = BiologicalAttention(window_size=(128, 128), center_bias=0.7)

    # Create model
    input_dim = 128 * 128 * 3
    output_dim = 128 * 128
    model = DualTetrahedralNetwork(input_dim, output_dim, latent_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    learner = BiologicalVisionLearner(model, optimizer, window_size=3)

    print(f"🧠 System: {device}")
    print(f"   Window: {attention.window_size}")
    print(f"   Center bias: 70%")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    attention.set_frame_size((frame_width, frame_height))

    print(f"📹 Video: {frame_width}×{frame_height}")
    print()

    # Track
    position_history = deque(maxlen=50)
    frame_idx = 0

    # Process frames
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert for learning
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_norm = frame_gray.astype(np.float32) / 255.0

        # Get attention position (NON-LINEAR)
        position = attention.next_position()
        position_history.append(position)

        # Extract window
        window = attention.extract_window(frame_gray_norm, position)

        # Learn
        if window.shape[:2] == attention.window_size[::-1]:
            loss = learner.observe_frame(window)
        else:
            loss = None

        # Visualize periodically
        if frame_idx % update_every == 0:
            clear_output(wait=True)

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: Full frame with attention
            display_frame = frame.copy()

            # Draw trail
            if len(position_history) > 1:
                points = list(position_history)
                for i in range(len(points) - 1):
                    p1 = (points[i][0] + attention.window_size[0]//2,
                          points[i][1] + attention.window_size[1]//2)
                    p2 = (points[i+1][0] + attention.window_size[0]//2,
                          points[i+1][1] + attention.window_size[1]//2)
                    alpha = i / len(points)
                    color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))
                    cv2.line(display_frame, p1, p2, color, 2)

            # Draw attention window
            x, y = position
            win_w, win_h = attention.window_size
            cv2.rectangle(display_frame, (x, y), (x + win_w, y + win_h), (0, 255, 0), 3)

            # Draw center marker
            center_x = (frame_width - win_w) // 2 + win_w // 2
            center_y = (frame_height - win_h) // 2 + win_h // 2
            cv2.drawMarker(display_frame, (center_x, center_y),
                          (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            # Convert BGR to RGB for matplotlib
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            axes[0].imshow(display_frame)
            axes[0].set_title(f'Biological Vision (Frame {frame_idx})', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            # Right: Attention window
            window_display = (window * 255).astype(np.uint8)
            axes[1].imshow(window_display, cmap='gray')
            axes[1].set_title(f'Attention Window\nPos: {position}', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # Add info
            info_text = f"""
Frame: {frame_idx}/{max_frames}
Updates: {learner.total_updates}
Loss: {loss:.6f if loss else 'N/A'}
Avg Loss: {np.mean(list(learner.loss_history)[-100:]):.6f if len(learner.loss_history) > 0 else 'N/A'}

GREEN BOX = Where model is looking
TRAIL = Saccadic eye movements
WHITE CROSS = Center (fovea)

Time flows LINEARLY (frame-by-frame)
Space is NON-LINEAR (saccades)
            """

            plt.figtext(0.5, 0.02, info_text.strip(), ha='center', fontsize=10,
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.show()

        frame_idx += 1

    cap.release()

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print(f"Frames: {frame_idx}")
    print(f"Updates: {learner.total_updates}")
    print(f"Final loss: {learner.current_loss:.6f}")
    print("=" * 70)


# ============================================================================
# STEP 7: RUN IT!
# ============================================================================

print("\n" * 2)
print("=" * 70)
print("🧠 BIOLOGICAL VISION - COLAB DEMO")
print("=" * 70)
print()

# Create test video
video_path = create_test_video("test_video.mp4", duration_seconds=3)
print()

# Run biological vision with visualization
visualize_biological_vision_colab(video_path, max_frames=90, update_every=10)

print()
print("🧠 This is how biology learns from vision!")
print("   Time: LINEAR (tick → tock)")
print("   Space: NON-LINEAR (saccades, center-biased)")
print()
