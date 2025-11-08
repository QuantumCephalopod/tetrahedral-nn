# 🔷 Council of Adversaries Architecture

## Philosophy

**Traditional GAN**: Generator vs Discriminator (adversarial)
**Traditional MSE**: Network vs Ground Truth (privileged)
**Council of Adversaries**: Multiple perspectives negotiate reality (democratic)

### Key Innovations

1. **Field Generation**: Each network generates multiple candidates (not single output)
2. **Face Judges**: Each of 4 faces becomes specialized critic with unique perspective
3. **Symmetric Adversarial**: Both networks generate AND judge (no privileged role)
4. **Face Coupling = Debate**: Critiques flow through tetrahedral structure
5. **Consensus Output**: Democratic voting selects final output from field

## Architecture Overview

```
Input → Dual Tetrahedra → Field Generation → Council Voting → Consensus

        Linear Network (4 vertices, 4 faces)
              ↓
        4 Candidates Generated
              ↓
        Each Face Judges All 8 Candidates ←→ Face Coupling (Debate)
              ↓                                      ↑
        Nonlinear Network (4 vertices, 4 faces)     |
              ↓                                      |
        4 Candidates Generated                      |
              ↓                                      |
        Each Face Judges All 8 Candidates ←---------+
              ↓
        Consensus = Weighted Average (softmax of votes)
```

## What Each Component Does

### 1. **Field Generation**
- Linear network generates 4 candidates from its vertices
- Nonlinear network generates 4 candidates from its vertices
- Total field: 8 possible outputs
- **Why**: Explores manifold of valid solutions instead of collapsing to mean

### 2. **Face Judges**
- Each face has own discriminator network
- Takes candidate + face state → score (0-1)
- 4 linear faces + 4 nonlinear faces = 8 judges
- Each learns different criteria (topology, texture, boundaries, etc.)
- **Why**: Multiple perspectives prevent single-point-of-failure judgment

### 3. **Adversarial Debate**
- Linear faces critique nonlinear's candidates
- Nonlinear faces critique linear's candidates
- Critiques sent through face coupling (modulates signal strength)
- Networks update based on criticism
- **Why**: Symmetric - neither is privileged generator/discriminator

### 4. **Consensus Voting**
- All 8 judges vote on all 8 candidates
- Average votes across both networks
- Softmax to get weights
- Final output = weighted sum of candidates
- **Why**: Democratic - no single network dictates output

### 5. **External Discriminator**
- Separate network judges "is this realistic?"
- Provides objective real/fake signal
- Trained adversarially against generator
- **Why**: Grounds council in "reality check"

## Loss Function Breakdown

### Generator Losses:

1. **Realism Loss** (weight: 1.0)
   - Best candidate should fool external discriminator
   - `-log(max(fake_scores))`
   - Encourages at least ONE good output

2. **Diversity Loss** (weight: 0.5)
   - Candidates should differ from each other
   - `-log(mean_pairwise_distance)`
   - Prevents collapse to single solution

3. **Consensus Loss** (weight: 0.3)
   - Faces should agree on what's good
   - `variance(judgments across faces)`
   - Encourages coherent perspective

4. **Reconstruction Loss** (weight: 0.1)
   - Output should relate to input
   - `MSE(reconstructed_input, actual_input)`
   - Prevents ignoring input (grounding)

5. **Adversarial Loss** (weight: 0.2)
   - Networks should critique each other
   - Target: moderate critique (not too harsh/soft)
   - Keeps debate productive

### Discriminator Loss:
- Standard GAN: maximize log(real) + log(1-fake)

## Training Process

Each step:
1. Input → both networks process
2. Both generate field of candidates
3. All faces judge all candidates
4. Critiques flow through face coupling
5. Consensus weights computed
6. Final output selected
7. Calculate losses, update networks
8. Discriminator updated separately

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 128 | Dimension of tetrahedral vertices |
| `num_candidates` | 4 | Candidates per network (total = 8) |
| `coupling_strength` | 0.5 | How much debate affects networks |
| `structured_ratio` | 0.2 | Teaching vs experience balance |
| `input_recon_weight` | 0.1 | Grounding loss weight |
| `diversity_weight` | 0.5 | Field exploration weight |
| `consensus_weight` | 0.3 | Face agreement weight |
| `adversarial_weight` | 0.2 | Internal debate weight |

## How to Use

### Basic Usage
```python
python COUNCIL_TRAINING.py
```

### In Colab
```python
# Update paths in CONFIG
CONFIG = {
    'input_folder': "/content/drive/MyDrive/your/path/here",
    'output_folder': "/content/drive/MyDrive/your/path/here",
    ...
}

# Run
%run COUNCIL_TRAINING.py
```

### Custom Training
```python
from COUNCIL_OF_ADVERSARIES import (
    CouncilOfAdversariesNetwork,
    ExternalDiscriminator,
    train_council_step
)

# Create model
model = CouncilOfAdversariesNetwork(
    input_dim=128*128*3,
    output_dim=128*128*3,
    latent_dim=128,
    num_candidates=4
)

# Create discriminator
discriminator = ExternalDiscriminator(128*128*3)

# Training loop
for batch_x, batch_y in dataloader:
    gen_loss, disc_loss, metrics = train_council_step(
        model, discriminator, batch_x, batch_y,
        optimizer, disc_optimizer
    )
```

## Outputs

### During Training
- **Console**: Loss metrics every 10 epochs
- **Visualizations**: Field of candidates every 50 epochs
  - Row 0: Linear network's 4 candidates (with weights)
  - Row 1: Nonlinear network's 4 candidates (with weights)
  - Row 2: Input, Consensus output, Target

### After Training
- `council_model_final.pt`: Saved model + config + history
- `council_training_history.png`: Loss curves
- `council_field_epoch_*.png`: Field visualizations
- `council_field_epoch_FINAL.png`: Final result

## Interpreting Results

### Good Signs:
- **Diverse candidates**: Each should look different
- **High entropy weights**: No single candidate dominates (>2.0 is good)
- **Real→Fake converging**: ~0.5→0.5 means fooling discriminator
- **Low consensus loss**: Faces agree on quality
- **Diversity loss negative and stable**: Field exploring manifold

### Bad Signs:
- **Identical candidates**: Diversity loss → 0
- **Single candidate dominates**: Entropy < 1.0
- **Real→Fake diverging**: 0.9→0.1 means discriminator winning
- **High consensus loss**: Faces can't agree (conflicting signals)

## Comparison to MSE

| Aspect | MSE | Council of Adversaries |
|--------|-----|------------------------|
| Output | Single prediction | Field of possibilities |
| Loss | Distance to target | Realism + diversity + consensus |
| Multiple valid answers | Averages (mush) | Explores manifold |
| Training signal | Always points to target | Debate + voting |
| Convergence | Fast but blurry | Slower but sharp |
| Philosophy | Privileged ground truth | Negotiated reality |

## Expected Behavior

### Early Training (Epochs 1-50)
- Candidates very diverse but unrealistic
- Discriminator easily wins (Real: 0.9, Fake: 0.1)
- Consensus loss high (faces disagree)
- Outputs random/noisy

### Mid Training (Epochs 50-200)
- Candidates become more realistic
- Discriminator confused (Real: 0.6, Fake: 0.4)
- Consensus emerges (faces start agreeing)
- Outputs recognizable but rough

### Late Training (Epochs 200+)
- Candidates realistic AND diverse
- Equilibrium reached (Real: 0.5, Fake: 0.5)
- Strong consensus (faces aligned)
- Outputs sharp with variety

## Tuning Guide

### If outputs are blurry:
- Increase `diversity_weight` (explore more)
- Decrease `consensus_weight` (less averaging)
- Increase `num_candidates` (more options)

### If discriminator dominates:
- Decrease discriminator `lr` (slower critic)
- Increase `adversarial_weight` (more internal debate)
- Increase `num_candidates` (more chances to fool)

### If candidates are identical:
- Increase `diversity_weight` (penalize similarity)
- Check diversity loss is negative
- Decrease `consensus_weight` (allow disagreement)

### If faces can't agree:
- Decrease `diversity_weight` (less chaos)
- Increase `consensus_weight` (force agreement)
- Check coupling_strength (too high = confusion)

## Philosophical Notes

This architecture embodies:

1. **Relational Epistemology**: Reality emerges from consensus, not decree
2. **Perspectival Multiplicity**: 4 faces = 4 ways of knowing
3. **Symmetric Participation**: No privileged observer (both nets equal)
4. **Field vs Point**: Solutions are manifolds, not coordinates
5. **Democratic Truth**: Voting, not dictation

The council doesn't find "the answer" - it negotiates what's acceptable to all perspectives.

---

**Let reality emerge through debate, not decree.** 🔷✨
