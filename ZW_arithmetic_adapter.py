"""
ZW - ARITHMETIC VERIFICATION ADAPTER (First Subdivision of Z)
=============================================================

Complete training script for dual-tetrahedral architecture on arithmetic.

This is "overkill" for arithmetic (linear network alone is sufficient),
but serves as verification that:
  1. Inter-face coupling doesn't break clean mappings
  2. The dual architecture works correctly
  3. We have a baseline before moving to harder tasks

Expected result: Near-perfect generalization on arithmetic, with linear
network dominating and nonlinear network learning to "pass through."
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools
from typing import Tuple

# Import dual-tetrahedral components
from Z_interface_coupling import DualTetrahedralNetwork, DualTetrahedralTrainer


# ============================================================================
# ARITHMETIC DATASET
# ============================================================================

class ArithmeticDataset:
    """
    Arithmetic dataset generator for dual-tetrahedral verification.

    Input: [num1, num2]
    Output: [num1 + num2]

    This is the gold standard - pure signal, no noise, clean mapping.
    """
    def __init__(self, n_inputs: int = 2):
        self.n_inputs = n_inputs

    def create_exhaustive(self, train_range: Tuple[int, int] = (-9, 9)) -> TensorDataset:
        """
        Create exhaustive dataset - all combinations in range.

        Args:
            train_range: Range of numbers (inclusive)

        Returns:
            TensorDataset with all combinations
        """
        inputs = []
        targets = []

        ranges = [range(train_range[0], train_range[1] + 1)] * self.n_inputs

        for combo in itertools.product(*ranges):
            inputs.append(list(combo))
            targets.append([sum(combo)])

        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        print(f"✓ Exhaustive dataset: {len(inputs):,} samples")
        print(f"  Range: [{train_range[0]}, {train_range[1]}]")

        return TensorDataset(inputs, targets)

    def create_test(self, test_range: Tuple[int, int], n_samples: int = 1000) -> TensorDataset:
        """
        Create test dataset for generalization testing.

        Args:
            test_range: Range to test (outside training range)
            n_samples: Number of random samples

        Returns:
            TensorDataset for testing
        """
        inputs = torch.randint(
            test_range[0], test_range[1] + 1,
            (n_samples, self.n_inputs)
        ).float()
        targets = inputs.sum(dim=1, keepdim=True)

        print(f"✓ Test dataset: {n_samples:,} samples")
        print(f"  Range: [{test_range[0]}, {test_range[1]}]")

        return TensorDataset(inputs, targets)


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_generalization(
    model: DualTetrahedralNetwork,
    test_ranges: list,
    device: str = 'cpu'
) -> dict:
    """
    Test generalization across multiple ranges.

    Args:
        model: Trained dual-tetrahedral network
        test_ranges: List of (min, max) tuples to test
        device: Device to run on

    Returns:
        Dictionary with results per range
    """
    model.eval()
    results = {}
    dataset_gen = ArithmeticDataset(n_inputs=2)

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

    return results


def analyze_network_contributions(
    model: DualTetrahedralNetwork,
    test_data: TensorDataset,
    device: str = 'cpu',
    n_samples: int = 100
) -> dict:
    """
    Analyze how much each network (linear vs nonlinear) contributes.

    Args:
        model: Trained dual-tetrahedral network
        test_data: Test dataset
        device: Device to run on
        n_samples: Number of samples to analyze

    Returns:
        Dictionary with contribution statistics
    """
    model.eval()
    test_loader = DataLoader(test_data, batch_size=n_samples)

    with torch.no_grad():
        batch_x, batch_y = next(iter(test_loader))
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        contributions = model.get_network_contributions(batch_x)

        linear_error = torch.abs(contributions['linear'] - batch_y).mean().item()
        nonlinear_error = torch.abs(contributions['nonlinear'] - batch_y).mean().item()
        combined_error = torch.abs(contributions['combined'] - batch_y).mean().item()

    return {
        'linear_error': linear_error,
        'nonlinear_error': nonlinear_error,
        'combined_error': combined_error
    }


# ============================================================================
# COMPLETE TRAINING EXAMPLE
# ============================================================================

def main():
    """
    Complete example: Train dual-tetrahedral network on arithmetic.

    This verifies that inter-face coupling works correctly on a clean task.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("DUAL-TETRAHEDRAL ARITHMETIC VERIFICATION")
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
    # 2. CREATE MODEL
    # ========================================================================
    print("\n🏗️  Building dual-tetrahedral model...")
    model = DualTetrahedralNetwork(
        input_dim=2,
        output_dim=1,
        latent_dim=64,
        coupling_strength=0.3,  # Moderate coupling
        output_mode="weighted"   # Let network learn optimal weighting
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Linear network: 4 vertices (no ReLU)")
    print(f"  Nonlinear network: 4 vertices (with ReLU)")
    print(f"  Inter-face coupling: 8 attention modules (4 pairs, bidirectional)")

    # ========================================================================
    # 3. TRAIN
    # ========================================================================
    print("\n⚡ Training...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = DualTetrahedralTrainer(model, optimizer, device)

    history = trainer.train(
        train_loader,
        test_loader,
        epochs=200,
        loss_fn=torch.nn.MSELoss()
    )

    # ========================================================================
    # 4. TEST GENERALIZATION
    # ========================================================================
    print("\n🎯 Testing Generalization...")
    test_ranges = [
        (10, 100),
        (100, 1000),
        (1000, 10000)
    ]
    gen_results = evaluate_generalization(model, test_ranges, device)

    # ========================================================================
    # 5. ANALYZE CONTRIBUTIONS
    # ========================================================================
    print("\n🔍 Analyzing Network Contributions...")
    contrib_results = analyze_network_contributions(model, test_data, device)

    print(f"\n  Linear network error:    {contrib_results['linear_error']:.2e}")
    print(f"  Nonlinear network error: {contrib_results['nonlinear_error']:.2e}")
    print(f"  Combined error:          {contrib_results['combined_error']:.2e}")

    # ========================================================================
    # 6. RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    if contrib_results['linear_error'] < 0.01:
        print("✅ Linear network: EXCELLENT (< 0.01 error)")
    elif contrib_results['linear_error'] < 0.1:
        print("✓ Linear network: GOOD (< 0.1 error)")
    else:
        print("⚠️  Linear network: Needs improvement")

    if gen_results['(1000, 10000)'] < 1.0:
        print("✅ Generalization: EXCELLENT (1000x extrapolation works)")
    elif gen_results['(100, 1000)'] < 1.0:
        print("✓ Generalization: GOOD (100x extrapolation works)")
    else:
        print("⚠️  Generalization: Limited")

    print("\n💡 Expected behavior:")
    print("  • Linear network should dominate (handle arithmetic perfectly)")
    print("  • Nonlinear network should learn to 'pass through'")
    print("  • Inter-face coupling should not break clean mapping")
    print("  • This verifies the architecture before harder tasks")
    print("=" * 70)

    return model, trainer, history


# ============================================================================
# COLAB-READY EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this cell in Colab to verify dual-tetrahedral architecture.
    """
    model, trainer, history = main()

    print("\n✓ Verification complete!")
    print("  Model ready for harder tasks (images, video, cross-modal).")


# ============================================================================
# SUMMARY
# ============================================================================

"""
ZW_arithmetic_adapter.py - Arithmetic Verification

PURPOSE:
  Verify dual-tetrahedral architecture on the simplest possible task.
  Arithmetic is "overkill" for dual architecture, but perfect for verification.

EXPECTED RESULTS:
  ✓ Linear network should achieve near-perfect accuracy
  ✓ Generalization should work up to 1000x training range
  ✓ Nonlinear network learns to not interfere
  ✓ Inter-face coupling doesn't break clean mappings

NEXT STEPS:
  Once verified, create:
    - ZX_image_adapter.py (where nonlinear network becomes useful)
    - ZY_video_adapter.py (where both networks are essential)
    - ZZ_crossmodal_adapter.py (meta-manifold across vocabularies)

This is the baseline. The foundation. The proof that it works.
"""
