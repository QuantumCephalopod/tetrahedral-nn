# Timescale Experiments - Multi-Rate Temporal Processing

**Theme:** Hierarchical memory and learning across nested timescales

**Core Philosophy:** "The brain has no single clock - memory flows at many rates"

---

## Quick Summary

These experiments implement **power-law memory** through nested timescales, where different components update at different rates following the golden ratio (φ). Fast dynamics capture immediate patterns, slow dynamics capture long-term structure, creating a spectrum from milliseconds to permanence.

**Key Insight:** Using φ (most irrational number) for timescale ratios prevents resonance artifacts and creates natural hierarchical memory.

---

## Experiments

### 1. TRAINING_10_NESTED_TIMESCALES.py
**Location:** `TIMESCALE_EXPERIMENTS/TRAINING_10_NESTED_TIMESCALES.py`

**What:** Image transformation learning with 10 nested temporal fields + blended MSE→SSIM loss

**Key Features:**
- **10 temporal memory fields** decaying at φ intervals:
  - Fast: τ = 0.1 (immediate)
  - φ hierarchy: 0.1/φ, 0.1/φ², ..., 0.1/φ⁹
  - Slow: τ = 0.1/φ⁹ ≈ 0.00003 (near-permanent)
- **Blended loss schedule**: 100% MSE → 20% MSE / 80% SSIM
- **Multi-timescale optimization**: Learning rates at φ intervals
- **Image-to-image tasks**: Rotation, scaling, translation

**Mathematical Foundation:**
```python
PHI = 1.618034  # Golden ratio
decay_i = base_decay / (PHI ** i)  # i-th timescale

field_i = field_i * (1 - decay_i) + new_activity * decay_i
```

**Why φ?**
- Most irrational number (worst rational approximation)
- Prevents period-locking between timescales
- Natural scaling in biological systems (phyllotaxis, spirals)
- Fibonacci sequence emerges: F(n+1)/F(n) → φ

**When to Use:**
- Understanding multi-timescale memory architecture
- Image transformation tasks requiring temporal context
- Testing perceptual loss functions (SSIM vs MSE)

---

### 2. CONTINUOUS_LEARNING_SYSTEM.py
**Location:** `TIMESCALE_EXPERIMENTS/CONTINUOUS_LEARNING_SYSTEM.py`

**What:** Online learning system with catastrophic forgetting prevention via timescale separation

**Key Features:**
- **Fast learners**: Quickly adapt to new data (high learning rate)
- **Slow learners**: Preserve old knowledge (low learning rate)
- **Synaptic consolidation**: Important memories transition fast→slow
- **Replay buffer**: Interleave old and new data
- **No task boundaries**: Continuous stream learning

**Philosophy:** "Memory is not storage - it's selective persistence"

**Catastrophic Forgetting Solution:**
```python
# Fast path: Learn new quickly
fast_params: lr = 0.001

# Slow path: Preserve old structure
slow_params: lr = 0.001 / φ³ ≈ 0.00023

# Consolidation: Move important fast→slow
if importance(memory) > threshold:
    slow_params += fast_params * consolidation_rate
```

**When to Use:**
- Lifelong learning scenarios
- Online/streaming data
- Multi-task learning without task boundaries
- Understanding memory consolidation

---

## Theoretical Foundation

### Power-Law Memory

Traditional exponential decay:
```python
memory(t) = exp(-t/τ)  # Single timescale τ
```

Power-law via nested exponentials:
```python
memory(t) = Σ w_i × exp(-t/τ_i)  # Multiple timescales
# When τ_i spaced at φ intervals → approximates power-law
```

**Result:** Heavy-tailed forgetting curve (biological!)

### Golden Ratio Hierarchy

```
Timescale Ladder:
τ₀ = 0.1         (~10 frames)      [Vertex-level: immediate activity]
τ₁ = 0.1/φ       (~16 frames)      [Edge-level: relational patterns]
τ₂ = 0.1/φ²      (~26 frames)      [Face-level: contextual structure]
τ₃ = 0.1/φ³      (~42 frames)      [Body-level: task episodes]
...
τ₉ = 0.1/φ⁹      (~7000 frames)    [Hub-level: permanent memories]
```

**Span:** Milliseconds to permanence in continuous spectrum

### Synaptic Consolidation

```python
# Fast synapses: plastic, high turnover
fast_weight += lr_fast × gradient

# Slow synapses: stable, protected
slow_weight += lr_slow × gradient

# Consolidation: Important fast→slow
if replay_count > threshold:
    slow_weight += lr_consolidate × fast_weight
    fast_weight *= decay_factor
```

Inspired by:
- **Sleep consolidation** (hippocampus → cortex)
- **Complementary learning systems** (McClelland et al.)
- **Elastic weight consolidation** (DeepMind)

---

## Connection to Active Inference

The **Active Inference Atari** work directly builds on these timescale insights:

```python
# Multi-timescale memory in Z_interface_coupling.py:
self.fast_field = None      # τ = 0.1
self.medium_field = None    # τ = 0.1/φ
self.slow_field = None      # τ = 0.1/φ²
# Hub embeddings: τ → 0 (permanent)
```

**Plus:** Learning rates at φ intervals:
```python
optimizer = Adam([
    {'params': vertex_params, 'lr': base_lr},
    {'params': edge_params, 'lr': base_lr / φ},
    {'params': face_params, 'lr': base_lr / φ²},
    {'params': coupling_params, 'lr': base_lr / φ³}
])
```

**Integration:** Temporal hierarchy + developmental curriculum + curiosity = embodied learning

---

## Perceptual Loss Evolution

### MSE → SSIM Blending

Early training:
```python
loss = MSE(prediction, target)  # Pixel-wise accuracy
```

Later training:
```python
loss = 0.2 × MSE + 0.8 × SSIM   # Structural similarity
```

**Why Blend?**
- **MSE bootstraps:** Gets coarse structure right (stable gradients)
- **SSIM refines:** Captures perceptual quality (texture, edges)
- **Smooth transition:** No abrupt loss changes (φ-based schedule)

**SSIM Formula:**
```python
SSIM = (2μ₁μ₂ + C₁)(2σ₁₂ + C₂) / ((μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂))
```

Where:
- μ: Mean (brightness)
- σ²: Variance (contrast)
- σ₁₂: Covariance (structure)

**Perceptual alignment:** SSIM correlates with human judgment better than MSE

---

## Key Findings

1. **φ-based timescales** prevent resonance, create natural hierarchies
2. **Power-law memory** emerges from nested exponentials
3. **Blended MSE→SSIM** improves perceptual quality after bootstrap
4. **Synaptic consolidation** prevents catastrophic forgetting
5. **Multi-rate learning** balances plasticity and stability

---

## Open Questions

- Optimal number of timescales? (10 vs continuous spectrum)
- Adaptive timescale selection based on task?
- Connection to predictive coding hierarchies?
- How to determine consolidation threshold?
- φ vs other irrationals (√2, e, π)?

---

## Usage Example

```python
# Load nested timescale experiment
from EXPERIMENTS.TIMESCALE_EXPERIMENTS.TRAINING_10_NESTED_TIMESCALES import (
    NestedTimescaleNetwork,
    BlendedLossScheduler
)

# Create network with 10 timescales
model = NestedTimescaleNetwork(
    input_dim=128*128*3,
    latent_dim=128,
    n_timescales=10,
    base_decay=0.1
)

# Blended loss
scheduler = BlendedLossScheduler(
    bootstrap_steps=1000,  # Pure MSE
    total_steps=10000      # Transition to SSIM
)

for step in range(10000):
    mse_weight, ssim_weight = scheduler.get_weights(step)
    loss = mse_weight * mse + ssim_weight * ssim
```

---

## When to Revisit This Cluster

- Implementing temporal prediction tasks (video, time series)
- Designing memory systems (working memory → long-term)
- Catastrophic forgetting problems
- Perceptual quality optimization (images, audio)
- Biological neural network modeling
- Hierarchical reinforcement learning (temporal abstraction)

---

## Related Papers

**Multi-timescale Learning:**
- Yamins & DiCarlo (2016) - "Using goal-driven deep learning models to understand sensory cortex"
- Eliasmith et al. (2012) - "A large-scale model of the functioning brain" (Spaun)
- Kietzmann et al. (2019) - "Recurrence is required for object recognition"

**Catastrophic Forgetting:**
- Kirkpatrick et al. (2017) - "Overcoming catastrophic forgetting in neural networks" (EWC)
- Zenke et al. (2017) - "Continual learning through synaptic intelligence"
- Parisi et al. (2019) - "Continual lifelong learning with neural networks: A review"

**Perceptual Loss:**
- Wang et al. (2004) - "Image quality assessment: from error visibility to structural similarity" (SSIM)
- Johnson et al. (2016) - "Perceptual losses for real-time style transfer"

---

**Navigation:**
- Return to: `EXPERIMENTS/` (top level)
- Related: `CONSENSUS_EXPERIMENTS.md` (consensus at different timescales), `ACTIVE_INFERENCE_ATARI.py` (φ-based timescales + curriculum)
- Architecture: `Z_COUPLING/Z_interface_coupling.py` (3-field timescale memory)
