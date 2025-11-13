# Attention Curriculum: Developmental Learning for World Models

**Author:** Philipp Remy Bartholomäus
**Date:** November 9, 2025
**Philosophy:** "What is worth predicting?"

---

## The Core Insight

When training a world model on Atari games with random exploration, **most actions do nothing**. The signal is sparse, making it nearly impossible for the network to learn which paddle it controls, let alone predict game physics.

**Solution:** Mirror infant development - learn agency (self-control) before understanding the external world.

---

## The Developmental Curriculum

### Five Phases (Golden Ratio Intervals)

Progressive attention masking that shrinks at φ intervals, expanding the model's "field of view" as it masters each phase:

#### **Phase 1: Control** (Steps 0-500)
- **Mask:** 100% (entire left half of screen blacked out)
- **What's visible:** Only player's paddle (right side) and immediate area
- **Mode:** Difference prediction (predicts CHANGE, not state)
- **Learning:** "My buttons move MY paddle"
- **Why:** Zero confusion about which paddle responds to input!

#### **Phase 2: Interaction** (Steps 500-1000)
- **Mask:** 61.8% (1/φ of left half masked)
- **What's visible:** Player paddle + ball trajectory
- **Mode:** Difference prediction
- **Learning:** "Ball bounces off my paddle"
- **Why:** Causal relationship between action and environment

#### **Phase 3: Understanding** (Steps 1000-2000)
- **Mask:** 38.2% (1/φ² of left half masked)
- **What's visible:** Ball + partial opponent paddle
- **Mode:** Difference prediction
- **Learning:** "Opponent affects ball trajectory"
- **Why:** Social interaction / multi-agent understanding

#### **Phase 4: Integration** (Steps 2000-3000)
- **Mask:** 23.6% (1/φ³ of left half masked)
- **What's visible:** Almost complete game field
- **Mode:** State prediction (transitions here!)
- **Learning:** Complete game dynamics
- **Why:** Putting it all together

#### **Phase 5: Complete** (Steps 3000+)
- **Mask:** 0% (full screen!)
- **Mode:** State prediction
- **Learning:** Full world model + strategy
- **Why:** Ready for policy learning

---

## Difference vs State Prediction

### Difference Mode (Phases 1-3)
```python
target = next_frame - current_frame  # What CHANGED?
```

**Why this is crucial:**
- No action → All zeros (no change)
- Button press → Clear paddle movement
- Ball hit → Obvious trajectory change
- **Makes causality crystal clear!**

### State Mode (Phases 4-5)
```python
target = next_frame  # What IS the next state?
```

Normal world model prediction once attention has expanded.

---

## Implementation Details

### Attention Masking (Left-Right for Pong)

Pong is played with paddles on the **sides**, not top/bottom:
- **Player paddle:** Right side (usually)
- **Opponent paddle:** Left side
- **Ball:** Moves horizontally

```python
def apply_attention_mask(frame, mask_amount, player='right'):
    """
    Mask opponent's region to focus learning on controllable area.

    Args:
        mask_amount: 0.0 = full view, 1.0 = only player's half visible
        player: 'right' or 'left' (which side is agent's paddle)

    Returns:
        Masked frame (opponent's side blacked out proportionally)
    """
    opponent_region_width = width // 2
    mask_width = int(opponent_region_width * mask_amount)

    if player == 'right':
        # Black out left side (opponent)
        mask[:, :, :, :mask_width] = 0.0
    else:
        # Black out right side
        mask[:, :, :, -mask_width:] = 0.0

    return frame * mask
```

### Golden Ratio Intervals

Mask shrinks at **φ ≈ 1.618** intervals:

| Phase | Steps | Mask | What's Masked |
|-------|-------|------|---------------|
| 1 | 0-500 | 1.0 | 100% (full left half) |
| 2 | 500-1000 | 1/φ = 0.618 | 61.8% of left half |
| 3 | 1000-2000 | 1/φ² = 0.382 | 38.2% of left half |
| 4 | 2000-3000 | 1/φ³ = 0.236 | 23.6% of left half |
| 5 | 3000+ | 0.0 | No mask (full view) |

**Why φ?**
- Most irrational number → prevents resonance artifacts
- Natural constant (appears in phyllotaxis, spiral growth)
- Maximal incommensurability between phases
- Same ratio used for learning rates and memory timescales!

---

## Connection to Other Components

### 1. Multi-Timescale Memory (Power-Law)

The attention curriculum works synergistically with golden ratio memory:

```python
# Memory fields (what persists)
fast_field:   decay = 0.1       (~10 frames, immediate)
medium_field: decay = 0.1/φ     (~16 frames, relational)
slow_field:   decay = 0.1/φ²    (~26 frames, contextual)
hub_memory:   learned params    (permanent, never decays!)
```

**The connection:**
- Attention expands spatially (what you see)
- Memory expands temporally (what you remember)
- Together: developmental understanding of space AND time

### 2. Blended Loss (MSE→SSIM)

Similar bootstrap strategy:
- **MSE** bootstraps structure (just like difference mode makes causality clear)
- **SSIM** refines perception (just like state mode captures full dynamics)

Both follow the principle: **Start simple, add complexity gradually**

### 3. Nested Timescale Learning Rates

```python
Vertices:  LR = base_lr          # Fast (reactive)
Edges:     LR = base_lr/φ        # Medium
Faces:     LR = base_lr/φ²       # Slow
Coupling:  LR = base_lr/φ³       # Slowest (field/permanent hubs)
```

**The pattern:** Golden ratio hierarchies everywhere!
- Space (attention curriculum)
- Time (memory fields)
- Learning (parameter updates)

---

## Why This Works

### 1. Sparse Signal Problem

**Before curriculum:**
- Random actions → 90% do nothing
- Model can't tell which paddle is "mine"
- Drowning in irrelevant information (opponent's side)
- No clear causal signal

**After curriculum:**
- Phase 1: ONLY see my paddle → clear action-effect mapping
- Phase 2: ONLY see my interaction with ball → causal physics
- Phase 3+: Gradually add complexity as fundamentals are mastered

### 2. Embodied Cognition

This mirrors how **real brains** develop:

1. **Infants (0-3 months):** Learn to control their own limbs
2. **Babies (3-6 months):** Understand object permanence (ball exists)
3. **Toddlers (6-12 months):** Social interaction (other agents exist)
4. **Children (1+ years):** Full world model + theory of mind

**Same sequence!** The curriculum encodes millions of years of evolutionary wisdom.

### 3. Curriculum Learning Without Explicit Curriculum

Unlike traditional curriculum learning (easier→harder tasks), this is **attention-based**:
- Same task (predict next frame)
- Same environment (Pong)
- **Different information availability** (what's visible)

The curriculum is emergent from masking, not hand-designed task ordering!

---

## Visualization

The live visualization shows **5 panels**:

1. **Observer View:** Full game (what a human sees)
2. **Model Input (Masked):** What the network actually receives
3. **Predicted Next:** Model's prediction
4. **Actual Next:** Ground truth
5. **Prediction Error:** Absolute difference (heatmap)

**Why show both?**
- Observer view: Understand the full context
- Model input: See exactly what the network learns from
- Comparison: Track how attention expands over time

---

## Metrics Tracked

```python
{
    'mse': [...],              # Mean squared error
    'ssim': [...],             # Structural similarity
    'total': [...],            # Combined loss
    'mse_weight': [...],       # Blending ratio (MSE)
    'ssim_weight': [...],      # Blending ratio (SSIM)
    'mask_amount': [...],      # Attention mask % (0-1)
    'phase': [...],            # Curriculum phase name
    'difference_mode': [...]   # True/False (diff vs state)
}
```

**Track everything!** This lets you see the co-evolution of:
- Loss landscape (what's being optimized)
- Attention (what's visible)
- Prediction mode (difference vs state)
- Blending weights (MSE vs SSIM)

---

## Checkpointing

Curriculum state is saved in checkpoints:

```python
checkpoint = {
    'step_count': ...,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'buffer': ...,              # Full replay buffer
    'history': ...,             # All metrics
    'curriculum_phase': ...,    # Current phase name
    'mask_amount': ...          # Current mask amount
}
```

**Resume training seamlessly!** Load checkpoint and curriculum continues from exact same phase.

---

## Future Directions

### 1. Adaptive Curriculum

Instead of fixed step counts, adapt based on performance:
```python
if mse_loss < threshold:
    advance_to_next_phase()
```

### 2. Multi-Game Transfer

Train Phase 1 on multiple games → learn universal "agency"
Transfer to new games → faster Phase 2+ learning

### 3. Hierarchical Attention

Not just left-right masking - also:
- Temporal masking (how far back can model see?)
- Feature masking (what channels are visible?)
- Saliency masking (attention follows important regions)

### 4. Active Exploration

Instead of random policy, use curriculum to guide exploration:
- Phase 1: Actions that move my paddle
- Phase 2: Actions that hit the ball
- Phase 3+: Actions that score points

---

## Key Takeaways

1. **Attention curriculum solves sparse signal problem** - Bootstrap on easy (self-control), expand to hard (world model)

2. **Golden ratio everywhere** - Spatial attention, temporal memory, learning rates all use φ

3. **Difference prediction is crucial** - Makes causality obvious during bootstrap phases

4. **Mirrors infant development** - Self → interaction → world is universal learning sequence

5. **Emergent curriculum** - No task engineering, just information gating

6. **Synergistic with other techniques** - Works beautifully with power-law memory and blended loss

---

## References

**Developmental Psychology:**
- Piaget's stages of cognitive development
- Embodied cognition theory
- Motor babbling in infants

**Machine Learning:**
- Curriculum learning (Bengio et al., 2009)
- Attention mechanisms (Bahdanau et al., 2014)
- World models (Ha & Schmidhuber, 2018)

**Natural Constants:**
- Golden ratio in phyllotaxis (Vogel, 1979)
- Power-law forgetting (Ebbinghaus, 1885)
- Scale-free networks (Barabási, 1999)

---

**This is developmental learning through the lens of geometry and natural constants. The curriculum isn't designed - it's discovered.** 🌱
