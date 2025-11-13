# Consensus Experiments - Multi-Representation Learning

**Theme:** Testing consensus mechanisms across multiple representational bases (W/X/Y/Z tetrahedral perspectives)

**Core Philosophy:** "Truth emerges from agreement across diverse perspectives"

---

## Quick Summary

These experiments explore learning through **consensus loss** - minimizing disagreement between multiple tetrahedral networks viewing the same data. Instead of MSE to ground truth, networks align with each other, discovering shared structure.

**Key Insight:** When W, X, Y, Z all agree on a representation, it captures fundamental invariants rather than superficial patterns.

---

## Experiments

### 1. CONSENSUS_ONLY.py
**Location:** `CONSENSUS_EXPERIMENTS/CONSENSUS_ONLY.py`

**What:** Pure consensus training without ground truth MSE

**Key Features:**
- Four networks (W/X/Y/Z perspectives)
- Consensus loss = disagreement between network outputs
- No reference to actual labels during training
- Tests if consensus alone finds structure

**Result:** Discovers shared representations through self-organization

**When to Use:** Understanding pure consensus dynamics

---

### 2. CONSENSUS_BASIS_TRANSFORM.py
**Location:** `CONSENSUS_EXPERIMENTS/CONSENSUS_BASIS_TRANSFORM.py`

**What:** Consensus with explicit basis transformations

**Key Features:**
- W/X/Y/Z bases defined geometrically (tetrahedral vertices)
- Input transformed to each basis before network processing
- Consensus measured in transformed spaces
- Tests geometric vs learned transformations

**Philosophy:** "Does geometry guide learning or emerge from it?"

**When to Use:** Connecting geometric intuition to learned representations

---

### 3. CONSENSUS_BASIS_DIFFUSION.py
**Location:** `CONSENSUS_EXPERIMENTS/CONSENSUS_BASIS_DIFFUSION.py`

**What:** Consensus with iterative diffusion/refinement

**Key Features:**
- Multi-step consensus propagation
- Networks iteratively update toward agreement
- Diffusion dynamics (smooth convergence)
- Tests temporal consensus evolution

**Inspired By:** Message-passing neural networks, belief propagation

**When to Use:** Modeling consensus as dynamic process, not static loss

---

### 4. CONSENSUS_MULTIREP_TRAINING.py
**Location:** `CONSENSUS_EXPERIMENTS/CONSENSUS_MULTIREP_TRAINING.py`

**What:** MSE vs Consensus loss comparison on multi-representation dataset

**Key Features:**
- Dataset with multiple valid representations per input
- Hybrid loss: α × MSE + (1-α) × Consensus
- Tests philosophical hypothesis: consensus > ground truth
- Detailed metrics and ablations

**Central Question:** "When data has multiple valid interpretations, does consensus find better solutions than MSE?"

**When to Use:** Rigorous comparison of learning objectives

---

### 5. CONSENSUS_MULTIREP_TRAINING_FIXED.py
**Location:** `CONSENSUS_EXPERIMENTS/CONSENSUS_MULTIREP_TRAINING_FIXED.py`

**What:** Bug-fixed version of CONSENSUS_MULTIREP_TRAINING

**Changes:** Fixed tensor shape mismatches, improved training stability

**When to Use:** Production version of multi-rep consensus experiments

---

## Theoretical Foundation

### Consensus Loss Formula

```python
consensus_loss = Σ |network_i(x) - network_j(x)|²  for all pairs (i,j)
```

Encourages all networks to produce similar outputs despite different internal processing.

### Hybrid Training

```python
total_loss = α × MSE(output, target) + β × consensus_loss
```

Balances:
- **MSE term**: Task accuracy (external alignment)
- **Consensus term**: Internal agreement (self-consistency)

### Why It Works

1. **Regularization:** Consensus prevents overfitting to noise (noise is view-dependent, signal is view-invariant)
2. **Feature Discovery:** Agreement across views = fundamental structure
3. **Robustness:** Multi-view representations are harder to fool (adversarial robustness)
4. **Uncertainty:** Low consensus = high uncertainty (calibrated confidence)

---

## Related Concepts

- **Ensemble Methods:** Bagging, boosting, stacking (but with shared training)
- **Co-training:** Multi-view learning from unlabeled data
- **Cross-view Consistency:** Temporal consistency in video, multi-modal alignment
- **Consensus ADMM:** Distributed optimization via consensus

---

## Key Findings

1. Pure consensus finds structure without labels (unsupervised learning)
2. Hybrid consensus+MSE outperforms pure MSE on ambiguous tasks
3. Geometric basis transformations provide useful inductive bias
4. Diffusion-based consensus converges more smoothly than direct loss
5. Consensus naturally provides uncertainty estimates (disagreement = uncertainty)

---

## Open Questions

- Optimal consensus weight β schedule during training?
- How many perspectives needed? (4 tetrahedral vs 8 cubic vs continuous?)
- Can consensus replace labels entirely for some tasks?
- Connection to Bayesian model averaging?

---

## Usage Example

```python
# Load a consensus experiment
from EXPERIMENTS.CONSENSUS_EXPERIMENTS.CONSENSUS_MULTIREP_TRAINING_FIXED import ConsensusPerspectiveNetwork

# Create multi-perspective network
model = ConsensusPerspectiveNetwork(
    input_dim=784,
    hidden_dim=128,
    output_dim=10,
    n_perspectives=4
)

# Train with hybrid loss
for epoch in range(epochs):
    loss = alpha * mse_loss + beta * consensus_loss
    loss.backward()
```

---

## When to Revisit This Cluster

- Implementing multi-agent systems (consensus = communication)
- Uncertainty quantification needs (disagreement = confidence)
- Multi-modal learning (audio + vision consensus)
- Adversarial robustness research (consensus is harder to fool)
- Distributed learning (each node is a perspective)

---

**Navigation:**
- Return to: `EXPERIMENTS/` (top level)
- Related: `TIMESCALE_EXPERIMENTS.md` (temporal consensus), `ACTIVE_INFERENCE_ATARI.py` (different perspectives via curriculum)
