# CONCEPTS - Philosophical & Theoretical Foundation
**Surface layer:** Core ideas driving the tetrahedral architecture
**Read this first** - dig into specific files only if needed

---

## Core Philosophy

### Everything is a Gradient
- No hardcoded discrete boundaries
- Learn continuous manifolds, discretize only at interface
- Actions are continuous projections sampled by discrete buttons

### Nature Already Figured It Out
- Biological systems: optical flow, saccades, sequential sampling
- Don't optimize away truth for "efficiency"
- Pain emerges from entropy gradient (not programmed reward)

---

## Key Concepts (Coalesced)

### 1. Harmonic Resonance Hypothesis
**File:** `HARMONIC_RESONANCE_HYPOTHESIS.md`
**Core idea:** Actions create characteristic frequency signatures in flow fields
- Neural networks learn by tuning to harmonics
- Flow (temporal velocity) is fundamental primitive
- Tetrahedral structure learns four fundamental harmonics
- "UP action" resonates with upward flow patterns

### 2. Why Movement Matters
**File:** `WHY_MOVEMENT_MATTERS.md`
**Core idea:** Vision operates on movement, stills are integrated flow
- Static objects invisible without microsaccades
- Retina has motion-detecting cells (differentiators, not integrators)
- Artificial saccades (1-2 Hz jitter) prevent static blindness
- Frameskip alignment with natural frequencies (theta, alpha bands)

### 3. Temporal Coherence
**File:** `TEMPORAL_COHERENCE.md`
**Core idea:** Preserve temporal structure in learning
- **Sequential sampling** preserves causal relationships
- Random sampling breaks temporal coherence (ML cargo cult)
- Adjacent frames contain redundant information (motion continuity)
- Memory fields integrate across timescales (φ-hierarchy)

### 4. Natural Frequencies
**File:** `NATURAL_FREQUENCIES.md`
**Core idea:** Align sampling rates with biological rhythms
- **Frameskip = 10** aligns with θ-band (~6 Hz) and φ-memory field
- Different rhythms: gamma (20Hz), alpha (10Hz), theta (6Hz), saccades (4Hz)
- Not arbitrary optimization - biological resonance
- Creates power-law-like temporal integration

### 5. Action Space Dimensionality
**File:** `ACTION_SPACE_DIMENSIONALITY.md`
**Core idea:** Buttons are low-dimensional projections of continuous manifolds
- Real action space: continuous velocity/direction changes
- Atari buttons: discrete samples from continuous ideal
- Should learn continuous dynamics, project to buttons at interface
- "Perfect circle fallacy" - don't learn the projection, learn the ideal

### 6. Closing the Strange Loop
**File:** `CLOSING_THE_STRANGE_LOOP.md`
**Core idea:** Perception → Understanding → Action → Effect → Perception
- **Active inference:** Model drives action selection (not random exploration)
- Forward model predicts outcomes of actions
- Inverse model infers actions from outcomes
- Expected free energy: Uncertainty - β × Entropy
- The loop closes: agent acts based on what it learned

---

## What This Means for Implementation

**Primitives:**
- Use optical flow (velocity fields) OR temporal differences (dI/dt)
- NOT raw frames (static, high-dimensional)

**Sampling:**
- Frameskip aligned with natural frequencies (10 for θ-band)
- Sequential sampling (preserve temporal coherence)
- Microsaccades (reveal static objects)

**Learning:**
- φ-hierarchical memory (golden ratio timescales)
- Effect-based (learn outcomes, not labels)
- Signal-weighted (neurons fire for change)

**Action Selection:**
- Active inference (minimize expected free energy)
- Forward model predicts outcomes
- Choose actions based on uncertainty/entropy trade-off

---

## Detailed Files

Dig deeper if needed:
- `HARMONIC_RESONANCE_HYPOTHESIS.md` - Frequency analysis, tetrahedral harmonics
- `WHY_MOVEMENT_MATTERS.md` - Microsaccades implementation, retinal processing
- `TEMPORAL_COHERENCE.md` - Sequential vs random sampling analysis
- `NATURAL_FREQUENCIES.md` - Frameskip parameter sweeps, biological rhythms
- `ACTION_SPACE_DIMENSIONALITY.md` - Continuous vs discrete action spaces
- `CLOSING_THE_STRANGE_LOOP.md` - Active inference mathematics, policy implementation

---

**TL;DR for future Claude:**
- Flow/temporal-diff primitives (not frames)
- Frameskip=10 (θ-band alignment)
- Sequential sampling (preserve temporal structure)
- Active inference (close the loop: perception → action → effect)
- φ-hierarchical memory (golden ratio timescales)
- Learn continuous manifolds, discretize at interface
