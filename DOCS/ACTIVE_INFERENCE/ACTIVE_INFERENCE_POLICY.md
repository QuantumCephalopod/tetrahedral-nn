# Active Inference Policy: Closing the Strange Loop

**Author:** Philipp Remy Bartholomäus
**Date:** November 9, 2025
**Philosophy:** "What is worth exploring?"

---

## The Strange Loop Closes

```
Input → Understanding → Action → Effect → New Input
   ↑                                         ↓
   └─────────────────────────────────────────┘
```

**Before:** Random exploration → Model learns physics → But actions ignore model!

**Now:** Model predictions → Guide actions → Better data → Better model → Better actions...

The loop is **closed**. Actions are driven by curiosity, not randomness.

---

## The Core Principle

Traditional RL: **Maximize reward**
Active Inference: **Maximize learning**

### Expected Free Energy

For each possible action, compute:

```python
Expected Free Energy = Uncertainty - β × Entropy
```

Where:
- **Uncertainty**: How unsure is the model about this action's outcome?
- **Entropy**: How diverse/interesting is the predicted outcome?
- **β**: Exploration coefficient (curriculum-dependent)

**Select action that MINIMIZES expected free energy.**

---

## Why This Works

### Phase 1: Control (β = 0.1, high exploration)

```
Free Energy = Uncertainty - 0.1 × Entropy
```

High entropy → Negative bonus → **LOW free energy** → **PREFERRED**

**Seeks actions with diverse, unpredictable outcomes = EXPLORATION**

Example:
- Action A: Predicted outcome has high variance → High entropy → Low free energy ✓
- Action B: Predicted outcome is boring/uniform → Low entropy → High free energy ✗

The model **actively seeks** actions that produce surprising, informative outcomes!

### Phase 5: Complete (β = 0.01, low exploration)

```
Free Energy = Uncertainty - 0.01 × Entropy
```

Low uncertainty → **LOW free energy** → **PREFERRED**

**Seeks actions with certain, predictable outcomes = EXPLOITATION**

Example:
- Action A: Model is very confident about outcome → Low uncertainty → Low free energy ✓
- Action B: Model is uncertain → High uncertainty → High free energy ✗

The model **exploits** what it knows, selecting reliable actions.

---

## Implementation Details

### Action Selection Algorithm

```python
def select_action(state, mask_amount):
    """
    Query world model for all possible actions.
    Select action that maximizes learning opportunity.
    """
    free_energies = []

    for action in all_possible_actions:
        # 1. Predict outcome
        predicted_next = model(state, action)

        # 2. Compute uncertainty (variance of prediction)
        uncertainty = variance(predicted_next)

        # 3. Compute entropy (same as variance for simplicity)
        entropy = variance(predicted_next)

        # 4. Get current β (exploration coefficient)
        β = curriculum_scheduler.get_beta(step)

        # 5. Expected Free Energy
        free_energy = uncertainty - β × entropy

        free_energies.append(free_energy)

    # 6. Select action (minimize free energy)
    # Use softmax for stochastic selection
    probs = softmax(-free_energies / temperature)
    action = sample(probs)

    return action
```

### Key Features

1. **Curriculum-Aware**: β changes with developmental phase
   - Control phase: High β → Seek diverse outcomes
   - Complete phase: Low β → Seek certain outcomes

2. **Stochastic Selection**: Uses softmax with temperature
   - Temperature = 1.0: Standard softmax
   - Higher temperature: More random (useful for early training)
   - Lower temperature: More greedy (useful for evaluation)

3. **Attention-Aware**: Applies curriculum mask before prediction
   - Model only sees what curriculum allows
   - Predictions respect developmental constraints

---

## Comparison to Other Approaches

### Random Policy (Before)

```python
def select_action(state):
    return random.choice(actions)
```

**Pros:**
- Simple
- Unbiased exploration
- No computation cost

**Cons:**
- Ignores learned model!
- Wastes time on boring actions
- Never improves over time

### Active Inference Policy (Now)

```python
def select_action(state):
    # Query model for all actions
    # Select based on expected free energy
    return action_that_maximizes_learning
```

**Pros:**
- **Uses learned world model** (closes the loop!)
- Curiosity-driven exploration
- Adapts with curriculum (explore → exploit)
- Focuses on informative experiences

**Cons:**
- Computationally expensive (N predictions per action)
- Requires well-trained model for good exploration
- More complex to debug

---

## Curriculum Integration

The policy synergizes with other curriculum components:

### 1. Attention Curriculum (What can model see?)

- **Phase 1**: Mask = 100% → Model only sees own paddle
- **Phase 5**: Mask = 0% → Model sees full game

**Policy adapts:** Early actions focus on self-control, later actions consider opponent.

### 2. Free Energy β Schedule (How to explore?)

- **Phase 1**: β = 0.1 → High exploration (seek surprise!)
- **Phase 5**: β = 0.01 → Low exploration (seek certainty)

**Policy adapts:** Early actions prioritize novelty, later actions prioritize reliability.

### 3. Difference vs State Prediction (What to predict?)

- **Phase 1-3**: Predict CHANGE (difference mode)
- **Phase 4-5**: Predict STATE (normal mode)

**Policy adapts:** Predictions respect current prediction mode.

All three components evolve together, creating a **developmental trajectory** for learning.

---

## Mathematical Foundation (Karl Friston)

### Free Energy Principle

Biological systems minimize **variational free energy**:

```
F = E_q[log q(s) - log p(o,s)]
  = D_KL[q(s) || p(s|o)] - log p(o)
  ≈ Prediction Error - Entropy
```

Where:
- **q(s)**: Agent's beliefs about states
- **p(o,s)**: Joint distribution of observations and states
- **D_KL**: KL divergence (surprise)
- **log p(o)**: Model evidence (expected)

### Expected Free Energy (Planning)

For action selection, minimize **expected free energy**:

```
G(π) = E_q[log q(s|π) - log p(o,s|π)]
     ≈ Epistemic Value + Pragmatic Value
     ≈ Information Gain - Expected Reward
```

In our implementation:
- **No explicit reward** (pure active inference)
- **Only epistemic value** (information gain)
- **β modulates** epistemic drive (curiosity)

This is **curiosity-driven active inference** without reward signals!

---

## Metrics to Track

When visualizing, monitor:

```python
{
    'action_entropy': [...],        # How diverse are selected actions?
    'prediction_variance': [...],   # How uncertain is model?
    'free_energy_spread': [...],    # Range of free energies across actions
    'β': [...],                     # Current exploration coefficient
    'selected_action_freq': {...}   # Which actions are preferred?
}
```

**Expected patterns:**

- **Early phases**: High action entropy, diverse exploration
- **Later phases**: Low action entropy, focused exploitation
- **Free energy spread**: Should decrease as model improves

---

## Ablation Studies

Compare three conditions:

### 1. Random Policy (Baseline)

```python
trainer = ActiveInferenceTrainer(use_active_inference_policy=False)
```

Pure exploration, model has no influence on actions.

### 2. Active Inference (Temperature = 1.0)

```python
trainer = ActiveInferenceTrainer(
    use_active_inference_policy=True,
    policy_temperature=1.0
)
```

Standard active inference with stochastic selection.

### 3. Active Inference (Temperature = 0.1)

```python
trainer = ActiveInferenceTrainer(
    use_active_inference_policy=True,
    policy_temperature=0.1
)
```

Greedy active inference (nearly deterministic).

**Hypothesis:**
- Random: Slowest learning, most diverse data
- Active (T=1.0): Fastest learning, balanced exploration
- Active (T=0.1): Fastest convergence, risk of local minima

---

## Connection to Embodied Cognition

This mirrors how **infants** learn:

### 1. Motor Babbling (Phase 1, β = 0.1)

- Random actions to discover what's controllable
- **High exploration**: Try diverse movements
- **Goal**: Learn body schema (agency)

### 2. Object Play (Phase 2-3, β = 0.05)

- Targeted actions on interesting objects
- **Moderate exploration**: Test object physics
- **Goal**: Understand causality

### 3. Goal-Directed Action (Phase 4-5, β = 0.01)

- Precise actions toward objectives
- **Low exploration**: Use known strategies
- **Goal**: Achieve specific outcomes

**Same developmental trajectory!** The curriculum encodes millions of years of evolution.

---

## Performance Considerations

### Computational Cost

For each action selection:
- **Random policy**: O(1) (instant)
- **Active inference**: O(N × M) where:
  - N = number of actions (18 for Atari)
  - M = forward pass cost (~10ms)

**Total**: ~180ms per action (10-20 FPS gameplay)

### Optimization Strategies

1. **Action Pruning**: Only evaluate likely actions
2. **Batch Prediction**: Predict all actions in parallel
3. **Cached Predictions**: Reuse predictions for similar states
4. **Model Distillation**: Train fast policy network on active inference targets

---

## Future Directions

### 1. Multi-Step Planning

Instead of single-step free energy, plan **trajectories**:

```python
for action_sequence in possible_trajectories:
    total_free_energy = sum([
        free_energy(step) for step in trajectory
    ])
```

**Tree search** over action sequences (like MCTS but with free energy).

### 2. Hierarchical Actions

Learn **macro-actions** (action sequences):
- Low-level: Individual button presses
- High-level: "Move paddle up", "Hit ball left"

Policy operates at high level, reduce computation.

### 3. Counterfactual Reasoning

Compare **actual vs predicted** outcomes:

```python
surprise = actual_outcome - predicted_outcome
if surprise > threshold:
    # Model was wrong! Update beliefs aggressively
```

**Precision weighting**: Trust predictions more when model is confident.

### 4. Social Active Inference

In multi-agent games, model **opponent's policy**:

```python
opponent_belief = infer_opponent_strategy(history)
free_energy = uncertainty - β × entropy + γ × theory_of_mind_gain
```

Learn faster by predicting opponent behavior!

---

## Key Takeaways

1. **Active inference closes the strange loop** - Actions driven by model predictions

2. **Curiosity is formalized** - Minimize free energy = maximize learning

3. **Curriculum integration** - β modulates exploration across developmental phases

4. **No reward needed** - Pure epistemic drive (information gain)

5. **Biologically plausible** - Mirrors infant development and free energy principle

6. **Computationally expensive** - Trade-off between intelligence and speed

7. **Emergent behavior** - Complex exploration strategies emerge from simple principle

---

## Usage Examples

### Quick Start

```python
# Create trainer with active inference
trainer = ActiveInferenceTrainer(
    env_name='ALE/Pong-v5',
    use_active_inference_policy=True,
    policy_temperature=1.0
)

# Train with curiosity-driven exploration
trainer.train_loop(n_episodes=50)

# Visualize learned policy
visualize_world_model_live(trainer, n_steps=200)
```

### Compare to Random

```python
# Random baseline
trainer_random = ActiveInferenceTrainer(
    use_active_inference_policy=False
)
trainer_random.train_loop(n_episodes=50)

# Active inference
trainer_active = ActiveInferenceTrainer(
    use_active_inference_policy=True
)
trainer_active.train_loop(n_episodes=50)

# Compare learning curves
plot_comparison(trainer_random.history, trainer_active.history)
```

---

## References

**Active Inference:**
- Friston et al. (2015) - "Active Inference and Learning"
- Friston (2010) - "The Free-Energy Principle: A Unified Brain Theory?"
- Parr & Friston (2017) - "The Anatomy of Inference"

**Curiosity-Driven Learning:**
- Schmidhuber (1991) - "Curious Model-Building Control Systems"
- Pathak et al. (2017) - "Curiosity-Driven Exploration (ICM)"
- Burda et al. (2018) - "Exploration by Random Network Distillation"

**Embodied Cognition:**
- Piaget (1952) - "The Origins of Intelligence in Children"
- Thelen & Smith (1994) - "A Dynamic Systems Approach to Development"
- Varela et al. (1991) - "The Embodied Mind"

**World Models:**
- Ha & Schmidhuber (2018) - "World Models"
- Hafner et al. (2019) - "Dream to Control: Learning Behaviors by Latent Imagination"

---

**The strange loop is closed. The model now learns by doing, and does to learn.** 🌀
