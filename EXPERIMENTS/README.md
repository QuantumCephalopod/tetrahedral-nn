# EXPERIMENTS - Navigation Hub

**Philosophy:** "The tetrahedral architecture learns through experience - these are its training grounds"

---

## Directory Structure

```
EXPERIMENTS/
├── README.md  ← You are here (navigation hub)
│
├── 📁 ACTIVE_INFERENCE/                  [CURRENT WORK]
│   ├── ACTIVE_INFERENCE_ATARI.py        🌀 Main trainer
│   └── ACTIVE_INFERENCE_LIVE_VIZ.py     🌀 Live visualization
│
├── 📁 CONSENSUS_EXPERIMENTS/
│   ├── CONSENSUS_ONLY.py
│   ├── CONSENSUS_BASIS_TRANSFORM.py
│   ├── CONSENSUS_BASIS_DIFFUSION.py
│   ├── CONSENSUS_MULTIREP_TRAINING.py
│   └── CONSENSUS_MULTIREP_TRAINING_FIXED.py
│
├── 📁 TIMESCALE_EXPERIMENTS/
│   ├── TRAINING_10_NESTED_TIMESCALES.py
│   └── CONTINUOUS_LEARNING_SYSTEM.py
│
├── 📁 IMAGE_EXPERIMENTS/
│   └── IMAGE_TRANSFORM.py
│
└── 📁 TESTING_VALIDATION/
    ├── BASELINE_TEST.py
    ├── GENERALIZATION_TEST.py
    └── COUNCIL_TRAINING.py
```

**Note:** Experiment status documentation moved to `DOCS/STATUS_*.md` files for hierarchical memory.

---

## Current Work

### 🌀 Active Inference for Atari

**Location:** `ACTIVE_INFERENCE/`

**Files:**
- `ACTIVE_INFERENCE_ATARI.py` - Main trainer with curiosity-driven policy
- `ACTIVE_INFERENCE_LIVE_VIZ.py` - Real-time visualization during training

**Status:** 🚧 **ACTIVE DEVELOPMENT**

**What:** World model learning for Atari games using active inference (no reward signal!)

**Key Features:**
- Forward model: predicts (frame, action) → next_frame
- Active inference policy: selects actions to maximize learning
- Attention curriculum: developmental learning (control → interaction → world)
- Free energy principle: curiosity via entropy bonus
- Multi-timescale memory: φ-based hierarchical memory

**Quick Start:**
```python
from EXPERIMENTS.ACTIVE_INFERENCE.ACTIVE_INFERENCE_ATARI import ActiveInferenceTrainer

trainer = ActiveInferenceTrainer(
    env_name='ALE/Pong-v5',
    use_active_inference_policy=True  # Closes the strange loop!
)
trainer.train_loop(n_episodes=50)
```

**Documentation:**
- `DOCS/STATUS_ACTIVE_INFERENCE.md` - Current status checkpoint
- `DOCS/ATTENTION_CURRICULUM.md` - Developmental curriculum approach
- `DOCS/ACTIVE_INFERENCE_POLICY.md` - Curiosity-driven action selection
- `DOCS/INVERSE_MODEL_FAILURE_NOTES.md` - What NOT to do (failed attempts)

**Philosophy:** "What is worth exploring?" - Learn physics through prediction error, select actions that maximize learning opportunity

---

## Experiment Clusters (Organized by Theme)

### 📊 Quick Reference Table

| Cluster | Theme | Files | Key Concept | Status |
|---------|-------|-------|-------------|--------|
| **Consensus** | Multi-perspective learning | 5 files | Agreement across diverse views = robust features | Archive |
| **Timescale** | Temporal hierarchy | 2 files | φ-based power-law memory | Integrated into active inference |
| **Image** | Spatial transformations | 1 file | Controlled test bed for architecture | Archive |
| **Testing** | Validation & ablation | 3 files | Trust, but verify | Always relevant |

---

## 1. Consensus Experiments

**Status:** `DOCS/STATUS_CONSENSUS.md`

**Core Idea:** Multiple networks (W/X/Y/Z perspectives) learn by minimizing disagreement. Consensus = robust representation.

**Key Experiments:**
- Pure consensus training (no labels!)
- Consensus vs MSE comparison
- Basis transformations (geometric inductive bias)
- Diffusion-based consensus (iterative refinement)

**When to Use:**
- Multi-modal learning
- Uncertainty quantification (disagreement = confidence)
- Adversarial robustness
- Distributed/federated learning

**Mathematical Foundation:**
```python
consensus_loss = Σ |network_i(x) - network_j(x)|²
# Minimize disagreement → shared structure emerges
```

---

## 2. Timescale Experiments

**Status:** `DOCS/STATUS_TIMESCALE.md`

**Core Idea:** Memory operates at multiple timescales following golden ratio (φ). Fast dynamics for immediate patterns, slow dynamics for long-term structure.

**Key Experiments:**
- 10 nested timescales (φ hierarchy)
- Continuous learning with synaptic consolidation
- MSE→SSIM blended loss evolution

**When to Use:**
- Temporal prediction (video, time series)
- Catastrophic forgetting prevention
- Lifelong learning
- Multi-rate dynamics

**Mathematical Foundation:**
```python
PHI = 1.618034  # Golden ratio
decay_i = base_decay / (PHI ** i)
memory_i = memory_i * (1 - decay_i) + activity * decay_i
# Power-law memory emerges from nested exponentials
```

**Integration:** These insights are now built into active inference (see `Z_COUPLING/Z_interface_coupling.py`)

---

## 3. Image Experiments

**Status:** `DOCS/STATUS_IMAGE.md`

**Core Idea:** Learn geometric transformations (rotation, scaling, translation) as controlled test bed for architecture.

**Key Experiment:**
- Image transform prediction
- Curriculum learning (small → large transforms)
- Visualization tools

**When to Use:**
- Testing new architectures
- Controlled debugging environment
- Spatial reasoning evaluation
- Teaching demonstrations

**Connection:** Transform prediction = forward model (same as active inference!)

---

## 4. Testing & Validation

**Status:** `DOCS/STATUS_TESTING.md`

**Core Idea:** Rigorous baselines and ablations. Beautiful philosophy needs empirical grounding.

**Key Experiments:**
- Vanilla MLP baseline comparison
- Generalization to OOD data
- Council (ensemble) voting
- Statistical significance testing

**When to Use:**
- **Always!** Before making claims
- Validating architectural choices
- Debugging unexpected results
- Publication preparation

**Best Practice:**
```python
# ALWAYS compare to simple baseline
results = {
    'Vanilla_MLP': 90.0%,      # Baseline
    'Dual_Tetra': 92.3%,       # +2.3% improvement
}
# Report with error bars: 92.3 ± 1.2% (n=5 seeds)
```

---

## Experiment Lifecycle

### Stage 1: Exploration (Sandbox)
- Quick prototypes
- Messy code
- Multiple ideas tested
- **Location:** Top-level `EXPERIMENTS/`

### Stage 2: Validation (Testing)
- Baseline comparisons
- Statistical tests
- Clean implementation
- **Location:** `TESTING_VALIDATION/`

### Stage 3: Integration (Production)
- Best ideas incorporated into main architecture
- Documented and tested
- **Location:** Core codebase (e.g., `Z_COUPLING/`)

### Stage 4: Archive (Memory)
- Experiments moved to thematic clusters
- Summary MD files created
- **Location:** Cluster folders with `.md` summaries

---

## Navigation Guide

### Starting New Conversation?

**Read this order:**
1. `EXPERIMENTS/README.md` ← Navigation hub (you are here)
2. `DOCS/STATUS_*.md` files (checkpoint status of each cluster)
3. Individual experiment files (deep dive if needed)

**Tier 1 (Hub):** This file - high-level map
**Tier 2 (Status):** `DOCS/STATUS_*.md` - checkpoint documentation
**Tier 3 (Files):** Actual experiment code - full details

### Looking for Specific Topic?

- **Active inference status** → `DOCS/STATUS_ACTIVE_INFERENCE.md`
- **Curiosity/exploration** → `DOCS/ACTIVE_INFERENCE_POLICY.md`
- **Developmental learning** → `DOCS/ATTENTION_CURRICULUM.md`
- **Multi-perspective** → `DOCS/STATUS_CONSENSUS.md`
- **Temporal dynamics** → `DOCS/STATUS_TIMESCALE.md`
- **Spatial reasoning** → `DOCS/STATUS_IMAGE.md`
- **Validation** → `DOCS/STATUS_TESTING.md`
- **Failed attempts** → `DOCS/INVERSE_MODEL_FAILURE_NOTES.md`

### Adding New Experiment?

1. Create file in appropriate cluster subdirectory (or at top-level if exploratory)
2. Work until complete/stable
3. Update or create `DOCS/STATUS_*.md` checkpoint documentation
4. Update this `README.md` navigation hub
5. Archive or integrate into core codebase when mature

---

## Cross-Cutting Themes

### Golden Ratio (φ) Everywhere

- **Timescales:** decay_i = decay / φⁱ
- **Learning rates:** lr_i = lr / φⁱ
- **Curriculum:** mask shrinks at φ intervals
- **Memory:** 3 fields at φ timescales

**Why φ?** Most irrational number → prevents resonance → natural hierarchies

### Developmental Learning

- **Attention curriculum:** Focus shrinks (self → interaction → world)
- **Free energy β:** Explore (β=0.1) → Exploit (β=0.01)
- **Loss blend:** MSE (bootstrap) → SSIM (perceptual)

**Inspiration:** Infant development (motor babbling → object play → goal-directed)

### Consensus as Principle

- **Consensus experiments:** Multiple networks agree
- **Active inference:** Model predicts, policy executes, world confirms
- **Testing:** Multiple seeds/baselines converge on truth

**Philosophy:** Truth emerges from agreement across diverse perspectives

---

## Key Papers Referenced Across Experiments

### Active Inference & Free Energy
- Friston et al. (2015) - "Active Inference and Learning"
- Friston (2010) - "The Free-Energy Principle"
- Parr & Friston (2017) - "The Anatomy of Inference"

### Curiosity-Driven Learning
- Schmidhuber (1991) - "Curious Model-Building Control"
- Pathak et al. (2017) - "Curiosity-Driven Exploration (ICM)"
- Burda et al. (2018) - "Random Network Distillation"

### Multi-Timescale Learning
- Yamins & DiCarlo (2016) - "Goal-driven deep learning models"
- Eliasmith et al. (2012) - "Spaun: Large-scale functioning brain"

### Catastrophic Forgetting
- Kirkpatrick et al. (2017) - "Overcoming catastrophic forgetting (EWC)"
- Zenke et al. (2017) - "Continual learning through synaptic intelligence"

### Ensemble & Multi-View
- Caruana et al. (2004) - "Ensemble selection"
- Blum & Mitchell (1998) - "Combining labeled and unlabeled data"

---

## Statistics

```
Total Experiments: 11 Python files (1 removed)
Active Development: 2 files (ACTIVE_INFERENCE/)
Archived Clusters: 4 clusters
Status Documentation: 5 checkpoint files (DOCS/STATUS_*.md)
Lines of Code: ~2200 (experiments) + ~1200 (docs)
Hierarchical Organization: ✅ Python in EXPERIMENTS/, Docs in DOCS/
```

---

## Contact & Philosophy

**Project:** Tetrahedral Neural Networks
**Author:** Philipp Remy Bartholomäus
**Principle:** "The river flows where it must"

**Core Questions:**
- ~~"What matters?"~~ → "What is worth predicting?"
- "How does understanding emerge from interaction?"
- "Can curiosity replace reward?"
- "What is the geometry of thought?"

**Status:** Active research, open-ended exploration

---

## Quick Commands

```bash
# List all experiments
ls EXPERIMENTS/

# List active inference cluster
ls EXPERIMENTS/ACTIVE_INFERENCE/

# Read status checkpoints
cat DOCS/STATUS_ACTIVE_INFERENCE.md
cat DOCS/STATUS_CONSENSUS.md

# Run active inference
python EXPERIMENTS/ACTIVE_INFERENCE/ACTIVE_INFERENCE_ATARI.py

# Check git history
git log --oneline EXPERIMENTS/
```

---

**Last Updated:** November 11, 2025
**Version:** 2.0 (Hierarchical memory reorganization - Python in EXPERIMENTS/, Docs in DOCS/)
