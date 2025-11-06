# 🔺🔺 Google Colab Quick Start Guide

This guide provides ready-to-run code snippets for Google Colab.

## Setup

### Step 1: Clone the Repository

```python
# Run this in a Colab cell
!git clone https://github.com/QuantumCephalopod/tetrahedral-nn.git
%cd tetrahedral-nn
```

### Step 2: Install Dependencies

```python
# PyTorch is usually pre-installed in Colab, but just in case:
!pip install torch torchvision opencv-python
```

---

## Quick Examples

### 🎯 Example 1: Arithmetic Generalization Test (Simplest)

Copy-paste this entire cell into Colab:

```python
# This demonstrates the core achievement: 1000x arithmetic generalization
exec(open('GENERALIZATION_TEST.py').read())
```

**What this does:**
- Trains on [-9, 9] (361 samples)
- Tests on ranges up to 10,000
- Shows float-precision-limited generalization
- Takes ~5 minutes on Colab GPU

---

### 🔄 Example 2: Rotation Learning

```python
# Learns to rotate 2D points (ZX subdivision topology)
exec(open('BASELINE_TEST.py').read())
```

**What this does:**
- Compares single vs dual tetrahedron
- Tests on rotation transformations
- Shows linear vs nonlinear performance

---

### 📹 Example 3: Continuous Video Learning

```python
# Upload a video file first, then:
from CONTINUOUS_LEARNING_SYSTEM import create_continuous_learner

learner = create_continuous_learner(img_size=64)
learner.watch_video('your_video.mp4')
learner.save_checkpoint('checkpoint.pt')
```

**What this does:**
- Watches video frame-by-frame
- No epochs, just continuous observation
- Accumulates context across videos
- Can pause/resume learning

---

### 🚀 Example 4: Inference Showcase (Deployment)

```python
exec(open('INFERENCE_SHOWCASE.py').read())
```

**What this does:**
- Shows how to deploy trained models
- Inference examples with visualization
- Performance metrics

---

## Manual Training (Full Control)

### Dual-Tetrahedral Network (Linear + Nonlinear)

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import dimensional components
from W_geometry import EDGE_INDICES, FACE_INDICES
from X_linear_tetrahedron import LinearTetrahedron
from Y_nonlinear_tetrahedron import NonlinearTetrahedron
from Z_interface_coupling import DualTetrahedralNetwork
from ZW_arithmetic_adapter import ArithmeticAdapter
from tetrahedral_trainer import TetrahedralTrainer

# Create adapter
adapter = ArithmeticAdapter(n_inputs=2)
train_data = adapter.create_dataset(train_range=(-9, 9), exhaustive=True)
test_data = adapter.create_test_dataset(test_range=(10, 100), n_samples=1000)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# Create dual network
model = DualTetrahedralNetwork(
    input_dim=2,
    output_dim=1,
    latent_dim=64,
    coupling_strength=0.5,
    output_mode="weighted"
)

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = TetrahedralTrainer(model, optimizer)
history = trainer.train(train_loader, test_loader, epochs=200, device='cuda')

print(f"Final test loss: {history['test_losses'][-1]:.6f}")
```

---

### Single Linear Tetrahedron (X dimension only)

```python
from X_linear_tetrahedron import LinearTetrahedron

# Just the linear network (no ReLU)
model = LinearTetrahedron(input_dim=2, latent_dim=64, output_dim=1)

# Rest same as above...
```

---

### Single Nonlinear Tetrahedron (Y dimension only)

```python
from Y_nonlinear_tetrahedron import NonlinearTetrahedron

# Just the nonlinear network (with ReLU)
model = NonlinearTetrahedron(input_dim=2, latent_dim=64, output_dim=1)

# Rest same as above...
```

---

## Using Different Adapters

### Arithmetic (ZW adapter)

```python
from ZW_arithmetic_adapter import ArithmeticAdapter

adapter = ArithmeticAdapter(n_inputs=2)
train_data = adapter.create_dataset(train_range=(-9, 9), exhaustive=True)
```

### Rotation (ZX adapter)

```python
from ZX_rotation_adapter import RotationAdapter

adapter = RotationAdapter(n_points=100)  # 100 2D points per sample
train_data = adapter.create_dataset(n_samples=1000)
```

### Temporal (ZY adapter)

```python
from ZY_temporal_adapter import TemporalAdapter

adapter = TemporalAdapter(sequence_length=10, feature_dim=32)
train_data = adapter.create_dataset(n_sequences=1000)
```

---

## Visualization in Colab

### Plot Training History

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_losses'], label='Train')
plt.plot(history['test_losses'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['test_maes'])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.yscale('log')
plt.title('Test MAE')

plt.tight_layout()
plt.show()
```

### Visualize Generalization

```python
import numpy as np

# Test at different ranges
test_ranges = [(10, 100), (100, 1000), (1000, 10000)]

for test_range in test_ranges:
    test_data = adapter.create_test_dataset(test_range=test_range, n_samples=100)
    test_loader = DataLoader(test_data, batch_size=100)

    loss, mae = trainer.evaluate(test_loader)

    print(f"Range [{test_range[0]}, {test_range[1]}]:")
    print(f"  Loss: {loss:.6f}, MAE: {mae:.6f}")
```

---

## Saving and Loading Models

### Save Checkpoint

```python
# Save everything
trainer.save_checkpoint('my_model.pt', epoch=200, metrics=history)

# Or manually:
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history
}, 'manual_checkpoint.pt')
```

### Load Checkpoint

```python
# Load using trainer
trainer.load_checkpoint('my_model.pt')

# Or manually:
checkpoint = torch.load('manual_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
history = checkpoint['history']
```

---

## Downloading Results from Colab

```python
# Download checkpoint to your computer
from google.colab import files
files.download('my_model.pt')

# Download plot
plt.savefig('results.png', dpi=150, bbox_inches='tight')
files.download('results.png')
```

---

## Troubleshooting

### Out of Memory on GPU

```python
# Reduce batch size
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)  # was 256

# Or reduce latent_dim
model = DualTetrahedralNetwork(input_dim=2, output_dim=1, latent_dim=32)  # was 64
```

### Slow Training

```python
# Make sure you're using GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

trainer = TetrahedralTrainer(model, optimizer)
history = trainer.train(train_loader, test_loader, epochs=200, device=device)
```

### Import Errors

```python
# Make sure you're in the right directory
import os
print(os.getcwd())  # Should end with /tetrahedral-nn

# If not:
%cd tetrahedral-nn
```

---

## Next Steps

- **Read `EXPLORATIONS.md`** for philosophical foundations
- **Read `README.md`** for technical details
- **Experiment with different adapters** (ZW, ZX, ZY)
- **Try continuous learning** with your own videos
- **Compare linear vs nonlinear vs dual** architectures

---

## Pro Tips

1. **Start with `GENERALIZATION_TEST.py`** - it's the fastest way to see the magic
2. **Use GPU runtime** in Colab (Runtime → Change runtime type → GPU)
3. **Files are self-contained** - you can copy-paste individual files into cells
4. **Check the dimensional structure** - W (geometry), X (linear), Y (nonlinear), Z (coupling)
5. **Fractal naming** - When dimensions split, they follow WW/WX/WY/WZ pattern

Happy experimenting! 🔺🔺
