# Copy-Paste Into Google Colab Cells

Just create cells and paste the code. No imports needed!

---

## CELL 1: W - Geometry Foundation

Paste the entire content of `W_geometry.py` into this cell.

---

## CELL 2: X - Linear Tetrahedron

Paste the entire content of `X_linear_tetrahedron.py` into this cell.

---

## CELL 3: Y - Nonlinear Tetrahedron

Paste the entire content of `Y_nonlinear_tetrahedron.py` into this cell.

---

## CELL 4: Z - Dual Network

Paste the entire content of `Z_interface_coupling.py` into this cell.

---

## CELL 5: ZW - Arithmetic Adapter

Paste the entire content of `ZW_arithmetic_adapter.py` into this cell.

---

## CELL 6: Trainer

Paste the entire content of `tetrahedral_trainer.py` into this cell.

---

## CELL 7: Create Data & Train

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Create adapter
adapter = ArithmeticAdapter(n_inputs=2)
train_data = adapter.create_dataset(train_range=(-9, 9), exhaustive=True)
test_data = adapter.create_test_dataset(test_range=(10, 100), n_samples=1000)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# Create model
model = DualTetrahedralNetwork(
    input_dim=2, output_dim=1, latent_dim=64,
    coupling_strength=0.5, output_mode="weighted"
)

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = TetrahedralTrainer(model, optimizer)
history = trainer.train(train_loader, test_loader, epochs=200, device='cuda')

print(f"✓ Done! Final test loss: {history['test_losses'][-1]:.6f}")
```

---

## CELL 8: Test Generalization

```python
print("\n🎯 GENERALIZATION TEST:")
for min_val, max_val in [(10,100), (100,1000), (1000,10000)]:
    test_data = adapter.create_test_dataset(test_range=(min_val, max_val), n_samples=1000)
    test_loader = DataLoader(test_data, batch_size=256)
    loss, mae = trainer.evaluate(test_loader)
    print(f"[{min_val}, {max_val}]: MAE = {mae:.6f}")
```

---

That's it! The Colab is now completely standalone.
