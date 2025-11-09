# Testing & Validation - Benchmarks and Ablations

**Theme:** Controlled comparisons to validate architectural choices

**Core Philosophy:** "Trust, but verify - every philosophical claim needs empirical grounding"

---

## Quick Summary

This cluster contains **baseline tests** and **ablation studies** to validate that the tetrahedral architecture actually provides benefits over standard approaches. Without these, beautiful philosophy remains ungrounded speculation.

**Key Insight:** Always compare to simple baselines. If fancy architecture doesn't beat plain MLP, question the architecture, not the baseline.

---

## Experiments

### 1. BASELINE_TEST.py
**Location:** `TESTING_VALIDATION/BASELINE_TEST.py`

**What:** Compare dual-tetrahedral network to vanilla MLP baseline

**Architectures Tested:**
1. **Vanilla MLP**: Simple feedforward (no tetrahedral structure)
2. **Single Tetrahedron**: Linear OR nonlinear (not both)
3. **Dual Tetrahedral**: Full architecture (linear + nonlinear + coupling)

**Metrics:**
- Training loss convergence
- Validation accuracy
- Generalization gap
- Parameter efficiency (accuracy per parameter)
- Training time

**Tasks:**
- MNIST classification (simple)
- CIFAR-10 classification (complex)
- Synthetic XOR-like patterns (nonlinearity test)

**Purpose:** Establish that tetrahedral structure provides measurable benefit

**Expected Result:**
- Vanilla MLP: Decent performance, many parameters
- Single Tetrahedron: Worse than dual (missing half the computation)
- Dual Tetrahedral: Best accuracy/parameter ratio

---

### 2. GENERALIZATION_TEST.py
**Location:** `TESTING_VALIDATION/GENERALIZATION_TEST.py`

**What:** Test generalization to out-of-distribution data

**Protocol:**
1. Train on subset (e.g., MNIST 0-4)
2. Test on held-out (MNIST 5-9)
3. Measure transfer performance

**Tests:**
- **Compositional generalization**: Train on simple, test on complex
- **Interpolation**: Test within training distribution
- **Extrapolation**: Test outside training range
- **Adversarial robustness**: Small perturbations

**Architectures:**
- Baseline MLP
- Dual Tetrahedral
- Consensus Network (multiple perspectives)

**Hypothesis:** Multi-perspective architectures generalize better (consensus = robust features)

**Metrics:**
- Accuracy drop (train → test)
- Calibration (confidence vs accuracy)
- Adversarial robustness (ε required to fool)

---

### 3. COUNCIL_TRAINING.py
**Location:** `TESTING_VALIDATION/COUNCIL_TRAINING.py`

**What:** Multi-network "council" that votes on predictions

**Architecture:**
```python
Council = {
    'Dual_Tetrahedral': DualTetrahedralNetwork(),
    'Vanilla_MLP': VanillaMLP(),
    'Consensus_Network': ConsensusNetwork(),
    'Tiny_Ensemble': [SmallNet() for _ in range(4)]
}

prediction = majority_vote(council)
# or weighted_vote based on confidence
```

**Purpose:**
- Compare architectures in ensemble setting
- Test if tetrahedral provides complementary errors
- Uncertainty via disagreement

**Philosophy:** "No single architecture is optimal - councils of diverse perspectives outperform individuals"

**Metrics:**
- Ensemble accuracy (should beat any individual)
- Diversity (do networks make different errors?)
- Calibration (disagreement = uncertainty?)

---

## Testing Principles

### 1. Always Include Vanilla Baseline

**DON'T:**
```python
results = {
    'Dual_Tetra': 92.3%,
    'Dual_Tetra_v2': 93.1%,  # Winner!
}
```

**DO:**
```python
results = {
    'Vanilla_MLP': 91.5%,     # Baseline
    'Dual_Tetra': 92.3%,      # +0.8%
    'Dual_Tetra_v2': 93.1%,   # +1.6%
}
```

**Reason:** Without baseline, can't tell if improvements are real or just tuning.

### 2. Control for Parameters

Compare at **equal parameter count**:
```python
Vanilla_MLP: 1M params → 90% accuracy
Dual_Tetra:  1M params → 92% accuracy  # Fair comparison!
```

**NOT:**
```python
Vanilla_MLP: 1M params → 90%
Dual_Tetra:  5M params → 92%  # Unfair! More params = higher capacity
```

### 3. Multiple Random Seeds

**Always** report mean ± std across seeds:
```python
# BAD: Single run
accuracy = 92.3%

# GOOD: Multiple seeds
accuracy = 92.3 ± 1.2%  (n=5 seeds)
```

**Reason:** Neural net training is stochastic. One lucky seed ≠ better architecture.

### 4. Statistical Significance

Use **t-test** or **Wilcoxon test** to verify differences:
```python
from scipy.stats import ttest_ind

baseline_scores = [90.1, 90.5, 89.8, 90.3, 90.0]  # n=5 seeds
tetra_scores = [92.1, 92.5, 91.9, 92.3, 92.2]

t_stat, p_value = ttest_ind(baseline_scores, tetra_scores)

if p_value < 0.05:
    print("Statistically significant improvement!")
else:
    print("Difference might be noise")
```

---

## Common Pitfalls (and How We Avoid Them)

### Pitfall 1: "Overfitting to test set"

**Problem:** Tune hyperparameters using test set, then report test accuracy
**Solution:** 3-way split: train / validation / test
- Train: Update weights
- Validation: Tune hyperparameters
- Test: Report final results (touch ONCE)

### Pitfall 2: "Unfair comparison"

**Problem:** Give fancy model more training time / better optimizer
**Solution:** Identical training protocol for all architectures

### Pitfall 3: "Cherry-picking results"

**Problem:** Run 20 experiments, report the 1 where fancy model wins
**Solution:** Pre-register experiments, report all results

### Pitfall 4: "Ignoring compute cost"

**Problem:** Model is 2% more accurate but 10x slower
**Solution:** Report accuracy AND training time / inference speed

---

## Ablation Study Template

Standard ablation for tetrahedral architecture:

```python
ablations = {
    'Vanilla_MLP': MLP(),                    # Baseline
    'Linear_Only': SingleTetra(linear=True),
    'Nonlinear_Only': SingleTetra(linear=False),
    'No_Coupling': DualTetra(coupling=0.0),
    'Full_Model': DualTetra(coupling=0.5),   # Our model
}

# Train all with identical settings
for name, model in ablations.items():
    train(model, epochs=100, lr=1e-3, batch_size=64)
    results[name] = evaluate(model, test_set)

# Report
print_table(results)
```

**Expected Pattern:**
- Vanilla < Linear_Only ≈ Nonlinear_Only < No_Coupling < Full_Model

**If not:** Architecture doesn't help! Rethink design.

---

## Key Findings (from these experiments)

1. **Dual-tetrahedral > single tetrahedron** (linear+nonlinear complementarity)
2. **Coupling helps** (0.5 coupling > 0.0, but diminishing returns > 0.5)
3. **Parameter efficiency** gains are modest (~10-20%, not 2-3x)
4. **Generalization** slightly better (consensus effect), but not dramatic
5. **Training time** similar to baseline (coupling adds minimal overhead)

**Honest assessment:** Architecture is promising but not revolutionary. Benefits are real but incremental.

---

## When to Run These Tests

**Before:**
- Publishing results
- Making architectural claims
- Deploying to production

**After:**
- Major architecture changes
- Adding new components
- Finding surprising results (could be bug!)

**Frequency:**
- Baseline test: Every major milestone
- Generalization test: Before claiming robustness
- Ablation: When adding new architectural components

---

## Metrics to Track

### Classification Tasks

```python
metrics = {
    'train_acc': [...],
    'val_acc': [...],
    'test_acc': [...],
    'train_loss': [...],
    'val_loss': [...],
    'param_count': int,
    'train_time': float,
    'inference_time': float
}
```

### Regression Tasks

```python
metrics = {
    'train_mse': [...],
    'train_ssim': [...],
    'val_mse': [...],
    'val_ssim': [...],
    'generalization_gap': val_loss - train_loss
}
```

### Robustness

```python
metrics = {
    'clean_acc': float,
    'adversarial_acc': float,
    'ood_acc': float,
    'calibration_error': float,
    'prediction_entropy': [...]
}
```

---

## Usage Example

```python
from EXPERIMENTS.TESTING_VALIDATION.BASELINE_TEST import run_comparison

# Run full comparison
results = run_comparison(
    architectures=['vanilla', 'single_tetra', 'dual_tetra'],
    datasets=['mnist', 'cifar10'],
    n_seeds=5,
    epochs=100
)

# Print results table
print_results(results)

# Statistical tests
for dataset in results:
    baseline = results[dataset]['vanilla']
    ours = results[dataset]['dual_tetra']
    p_value = significance_test(baseline, ours)
    print(f"{dataset}: p={p_value:.4f}")
```

---

## Open Questions

- Is 10-20% improvement enough to justify complexity?
- Do benefits scale with model size? (tiny models vs large models)
- Task-dependent benefits? (vision vs NLP vs RL)
- How to measure "emergent properties" quantitatively?

---

## Related Methodologies

**Machine Learning:**
- Ablation studies (Melis et al. 2018 - "On the State of the Art of Evaluation in Neural Language Models")
- Benchmarking best practices (Sculley et al. 2018 - "Winner's Curse?")
- Statistical testing (Dror et al. 2018 - "The Hitchhiker's Guide to Testing Statistical Significance")

**Experimental Design:**
- A/B testing protocols
- Statistical power analysis
- Multiple comparison correction (Bonferroni, Benjamini-Hochberg)

---

## When to Revisit This Cluster

- Making claims about architectural superiority
- Need ammunition for paper reviews ("where's the baseline?")
- Debugging unexpected results (is fancy model actually helping?)
- Choosing between design alternatives (ablate to decide)
- Teaching scientific rigor in ML

---

**Navigation:**
- Return to: `EXPERIMENTS/` (top level)
- Related: All experiment clusters (these tests validate them)
- See: `DOCS/ATTENTION_CURRICULUM.md`, `DOCS/ACTIVE_INFERENCE_POLICY.md` (validated through experiments)
