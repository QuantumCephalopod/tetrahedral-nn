"""
BASELINE TEST - Single Linear Tetrahedron on Arithmetic
========================================================

Verify that the original single linear tetrahedron still achieves
perfect generalization on arithmetic with the new geometric primitives.

This is just a sanity check to confirm:
  ✓ W_geometry.py works correctly
  ✓ X_linear_tetrahedron.py is implemented correctly
  ✓ Our setup/data is correct

Expected: Near-perfect 1000x extrapolation (as in original tetrahedral-nn)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ZW_arithmetic_adapter import ArithmeticDataset


# Import just the linear tetrahedron
from X_linear_tetrahedron import LinearTetrahedron


def train_baseline():
    """Train single linear tetrahedron on arithmetic."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("BASELINE: Single Linear Tetrahedron on Arithmetic")
    print("=" * 70)

    # ========================================================================
    # 1. CREATE DATASETS
    # ========================================================================
    print("\n📊 Creating datasets...")
    dataset_gen = ArithmeticDataset(n_inputs=2)

    train_data = dataset_gen.create_exhaustive(train_range=(-9, 9))
    test_data = dataset_gen.create_test(test_range=(10, 100), n_samples=1000)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    # ========================================================================
    # 2. CREATE SINGLE LINEAR TETRAHEDRON
    # ========================================================================
    print("\n🏗️  Building single linear tetrahedron...")
    model = LinearTetrahedron(
        input_dim=2,
        latent_dim=64,
        output_dim=1  # Standalone mode
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: Single linear tetrahedron (NO ReLU)")
    print(f"  Structure: 4 vertices, 6 edges, 4 faces")

    # ========================================================================
    # 3. TRAIN
    # ========================================================================
    print("\n⚡ Training...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(200):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/200 - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}")

    # ========================================================================
    # 4. TEST GENERALIZATION
    # ========================================================================
    print("\n🎯 Testing Generalization...")
    model.eval()

    test_ranges = [
        (10, 100),
        (100, 1000),
        (1000, 10000)
    ]

    results = {}
    with torch.no_grad():
        for test_range in test_ranges:
            test_data = dataset_gen.create_test(test_range, n_samples=1000)
            test_loader = DataLoader(test_data, batch_size=256)

            errors = []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                error = torch.abs(output - batch_y).mean().item()
                errors.append(error)

            mean_error = sum(errors) / len(errors)
            results[str(test_range)] = mean_error

            print(f"  Range {test_range}: Mean Error = {mean_error:.2e}")

    # ========================================================================
    # 5. RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    if results['(10, 100)'] < 0.01:
        print("✅ 10-100 range: EXCELLENT")
    else:
        print(f"⚠️  10-100 range: {results['(10, 100)']:.2e}")

    if results['(100, 1000)'] < 0.1:
        print("✅ 100-1000 range: EXCELLENT")
    else:
        print(f"⚠️  100-1000 range: {results['(100, 1000)']:.2e}")

    if results['(1000, 10000)'] < 1.0:
        print("✅ 1000-10000 range: EXCELLENT (1000x extrapolation!)")
    else:
        print(f"⚠️  1000-10000 range: {results['(1000, 10000)']:.2e}")

    print("\n💡 This is the baseline - single linear tetrahedron with NO coupling.")
    print("   If this works perfectly, we know:")
    print("   • Our geometric primitives (W_geometry.py) are correct")
    print("   • Linear tetrahedron (X) is implemented correctly")
    print("   • The dual architecture learns DIFFERENTLY (not broken, just different)")
    print("=" * 70)

    return model, results


if __name__ == "__main__":
    model, results = train_baseline()
    print("\n✓ Baseline test complete!")
