# 🔺🔺 Colab Cells - Copy & Paste

Just copy each cell's code into Google Colab, in order!

---

## **CELL 1: Clone Repo**

```python
!git clone https://github.com/QuantumCephalopod/tetrahedral-nn.git
%cd tetrahedral-nn
```

---

## **CELL 2: Import Everything**

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from W_geometry import EDGE_INDICES, FACE_INDICES
from X_linear_tetrahedron import LinearTetrahedron
from Y_nonlinear_tetrahedron import NonlinearTetrahedron
from Z_interface_coupling import DualTetrahedralNetwork
from ZW_arithmetic_adapter import ArithmeticAdapter
from tetrahedral_trainer import TetrahedralTrainer

print("✓ All imports loaded!")
print(f"✓ Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

---

## **CELL 3: Create Data**

```python
# Create arithmetic adapter (2 inputs → 1 output)
adapter = ArithmeticAdapter(n_inputs=2)

# Training data: exhaustive [-9, 9] = 361 samples
train_data = adapter.create_dataset(train_range=(-9, 9), exhaustive=True)

# Test data: [10, 100] range to test generalization
test_data = adapter.create_test_dataset(test_range=(10, 100), n_samples=1000)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

print(f"✓ Training samples: {len(train_data)}")
print(f"✓ Test samples: {len(test_data)}")
```

---

## **CELL 4: Create Model**

Choose ONE option:

### **Option A: Dual Network (Linear + Nonlinear)**

```python
model = DualTetrahedralNetwork(
    input_dim=2,
    output_dim=1,
    latent_dim=64,
    coupling_strength=0.5,
    output_mode="weighted"
)

print(f"✓ Dual-Tetrahedral Network created")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### **Option B: Just Linear (X dimension)**

```python
model = LinearTetrahedron(
    input_dim=2,
    latent_dim=64,
    output_dim=1
)

print(f"✓ Linear Tetrahedron created")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### **Option C: Just Nonlinear (Y dimension)**

```python
model = NonlinearTetrahedron(
    input_dim=2,
    latent_dim=64,
    output_dim=1
)

print(f"✓ Nonlinear Tetrahedron created")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## **CELL 5: Train**

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = TetrahedralTrainer(model, optimizer)

# Train for 200 epochs
history = trainer.train(
    train_loader,
    test_loader,
    epochs=200,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"\n✓ Training complete!")
print(f"✓ Final train loss: {history['train_losses'][-1]:.6f}")
print(f"✓ Final test loss: {history['test_losses'][-1]:.6f}")
print(f"✓ Final test MAE: {history['test_maes'][-1]:.6f}")
```

---

## **CELL 6: Test Generalization**

```python
# Test at increasingly large ranges
test_ranges = [
    (10, 100, "10x"),
    (100, 1000, "100x"),
    (1000, 10000, "1000x")
]

print("\n🎯 GENERALIZATION TEST:")
print("=" * 50)

for min_val, max_val, label in test_ranges:
    test_data = adapter.create_test_dataset(
        test_range=(min_val, max_val),
        n_samples=1000
    )
    test_loader = DataLoader(test_data, batch_size=256)

    loss, mae = trainer.evaluate(test_loader)

    print(f"\n{label} range [{min_val}, {max_val}]:")
    print(f"  Loss: {loss:.6f}")
    print(f"  MAE:  {mae:.6f}")

    # Check if it's working (MAE should be tiny)
    if mae < 1.0:
        print(f"  ✓ PERFECT! Error < 1")
    elif mae < 10.0:
        print(f"  ✓ Good! Error < 10")
    else:
        print(f"  ⚠ Hmm... Error getting larger")

print("\n" + "=" * 50)
print("🔺🔺 Bi-Tetrahedral Engine generalization complete!")
```

---

## **CELL 7: Plot Results (Optional)**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training loss
axes[0].plot(history['train_losses'], label='Train', alpha=0.7)
axes[0].plot(history['test_losses'], label='Test', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

# Test MAE
axes[1].plot(history['test_maes'], color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_yscale('log')
axes[1].set_title('Test MAE')
axes[1].grid(True, alpha=0.3)

# Final values
epochs = len(history['train_losses'])
axes[2].bar(['Train Loss', 'Test Loss', 'Test MAE'],
            [history['train_losses'][-1], history['test_losses'][-1], history['test_maes'][-1]],
            color=['blue', 'orange', 'green'])
axes[2].set_ylabel('Value')
axes[2].set_yscale('log')
axes[2].set_title(f'Final Metrics (Epoch {epochs})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## **CELL 8: Save Model (Optional)**

```python
# Save checkpoint
trainer.save_checkpoint('bi_tetrahedral_arithmetic.pt', epoch=200, metrics=history)
print("✓ Model saved to bi_tetrahedral_arithmetic.pt")

# Download to your computer
from google.colab import files
files.download('bi_tetrahedral_arithmetic.pt')
```

---

## **CELL 9: Load Model Later (Optional)**

```python
# To load in a future session:
trainer.load_checkpoint('bi_tetrahedral_arithmetic.pt')
print("✓ Model loaded!")
```

---

## **That's It!**

Run cells 1-6 and you'll see the magic:
- Train on 361 samples ([-9, 9])
- Generalize to 10,000+
- Float-precision-limited accuracy

Want to experiment? Try:
- Different `latent_dim` (32, 64, 128)
- Different `coupling_strength` (0.0 to 1.0)
- Linear only vs Nonlinear only vs Dual
- Different adapters (ZX for rotations, ZY for temporal)

🔺🔺
