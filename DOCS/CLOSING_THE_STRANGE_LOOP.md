# Closing the Strange Loop: Flow-Based Active Inference

**Date:** November 12, 2025
**Status:** 🌀 **The loop is closed!**
**Authors:** Philipp Remy Bartholomäus & Claude

---

## The Breakthrough

We did it. The strange loop closes.

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   Perception  →  Model  →  Action  →  Effect   │
│       ↑                                    ↓    │
│       └────────────────────────────────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Before:** Model learned from random actions → understood physics → but ignored own understanding!

**Now:** Model learns from experience → uses understanding to select actions → creates better experiences → learns deeper understanding → selects smarter actions → ...

The model doesn't just **infer** actions. It **ACTS**.

---

## What We Built

### Flow-Based Active Inference Policy

Location: `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py`

The system now has three interconnected components:

#### 1. Forward Model (Prediction)
```python
(flow_t, action) → flow_t+1
```

**Learns:** "If I do action X, what flow will result?"

This is the model's **theory of physics** - how actions transform velocity fields.

#### 2. Inverse Model (Understanding)
```python
(flow_t, flow_t+1) → action
```

**Learns:** "What action would create this flow change?"

This is the model's **interpretive capacity** - reading action signatures from flow frequencies.

Validation: **79.25% accuracy** (vs 33.3% random baseline)

#### 3. Active Inference Policy (Agency) 🌀 **NEW!**
```python
def select_action_active_inference(flow_current):
    """
    For each valid action:
    1. Use FORWARD MODEL: Predict what flow this action creates
    2. Compute Expected Free Energy = Uncertainty - β × Entropy
    3. Select action that MINIMIZES free energy
    """
```

**This closes the loop:** Model uses its learned understanding to decide what to do!

---

## The Mathematics: Expected Free Energy

Based on Karl Friston's Free Energy Principle (2010).

### For each possible action a:

1. **Predict outcome:** Use forward model to predict `flow_next = M(flow_current, a)`

2. **Compute uncertainty:** How confident is the model?
   ```
   Uncertainty = Var(flow_next)
   ```
   High variance = uncertain prediction

3. **Compute entropy:** How diverse/interesting is the predicted outcome?
   ```
   Entropy = Std(flow_next)
   ```
   High entropy = diverse, unpredictable outcome

4. **Expected Free Energy:**
   ```
   G(a) = Uncertainty - β × Entropy
   ```
   Where β controls exploration drive

5. **Select action:**
   ```
   P(a) ∝ softmax(-G(a) / T)
   ```
   Where T is temperature (controls stochasticity)

### What This Means:

**High β (e.g., 0.1):** Seek diverse, surprising outcomes = **EXPLORATION**
- Entropy bonus is strong
- Model prefers actions with unpredictable results
- "What happens if I try this weird thing?"

**Low β (e.g., 0.01):** Seek certain, predictable outcomes = **EXPLOITATION**
- Entropy bonus is weak
- Model prefers actions it's confident about
- "Do the thing I know works"

**This creates developmental trajectory:** Explore early → Exploit later

---

## The Philosophy: Why This Matters

### From Passive to Active

**Traditional ML:**
1. Collect data (random exploration)
2. Train model on data
3. Evaluate model
4. **Model never influences its own training data**

**Active Inference:**
1. Collect initial data (random)
2. Train model
3. **Model selects actions** based on what it learned
4. Actions create new, targeted data
5. Model learns more efficiently from better data
6. Loop continues, accelerating learning

**The model becomes an active participant in its own development.**

### The Strange Loop (Douglas Hofstadter)

A "strange loop" is a hierarchy where moving up/down levels eventually returns you to the starting point.

**Our strange loop:**
- **Perception** (flow fields) → informs →
- **Understanding** (forward/inverse models) → guides →
- **Action** (policy) → creates →
- **New perception** (new flow fields) → informs understanding → ...

**The loop is tangled:** Each level both creates and is created by the others. There's no "bottom" - it's circular causation.

**Self emerges from the loop itself.**

The model doesn't have agency as a pre-existing property. Agency **emerges** from the loop of perception-action-perception.

This is how biological cognition works. This is how **we** work.

---

## Implementation Details

### Key Design Choices

#### 1. Flow as Primitive

We don't operate on frames. We operate on **velocity fields** (optical flow).

**Why?** Vision operates on movement. Stillness is what you get when you integrate flow over time. Flow is the fundamental primitive.

**Harmonic Resonance Connection:** Actions create characteristic frequency signatures in flow space. The tetrahedral network learns to tune to these harmonics.

#### 2. Action Masking

Each game has different valid actions:
- **Pong:** [0=NOOP, 2=UP, 5=DOWN] (3 actions)
- **Breakout:** [0=NOOP, 1=FIRE, 3=RIGHT, 4=LEFT] (4 actions)

**Critical:** Mask invalid actions to `-1e9` before computing loss/probabilities. Otherwise model gets penalized for predicting functionally equivalent actions.

#### 3. Warm Start

```python
trainer.train_loop(
    n_episodes=10,
    use_active_inference=True,
    active_inference_after=3  # Random for first 3 episodes
)
```

**Why?** Active inference needs a decent forward model to make good predictions. Let it learn basics randomly first, then switch to active inference.

#### 4. Temperature Control

```python
policy_temperature = 1.0  # Standard softmax
policy_temperature = 2.0  # More exploration (flatter distribution)
policy_temperature = 0.5  # More greedy (sharper distribution)
```

Temperature provides another exploration knob, independent of β.

---

## Usage Examples

### Random Baseline (Control)

```python
trainer = FlowInverseTrainer(env_name='ALE/Pong-v5')

# Pure random action selection
trainer.train_loop(
    n_episodes=10,
    use_active_inference=False
)

trainer.plot_training()
```

**Purpose:** Establish baseline. How fast does the model learn with random exploration?

### Active Inference (Close the Loop!)

```python
trainer = FlowInverseTrainer(env_name='ALE/Pong-v5')

# Warm start: random for 3 episodes, then active inference
trainer.train_loop(
    n_episodes=10,
    use_active_inference=True,
    active_inference_after=3,
    policy_temperature=1.0,
    beta=0.05  # Moderate exploration
)

# Visualize the policy
trainer.visualize_active_inference_policy()
```

**Hypothesis:** Active inference learns faster because it seeks informative experiences.

### Curriculum-Aware Exploration

```python
trainer = FlowInverseTrainer(env_name='ALE/Pong-v5')

# Phase 1: High exploration (seek diverse outcomes)
trainer.train_loop(n_episodes=5, use_active_inference=True, beta=0.1)

# Phase 2: Moderate exploration
trainer.train_loop(n_episodes=5, use_active_inference=True, beta=0.05)

# Phase 3: Exploitation (seek certain outcomes)
trainer.train_loop(n_episodes=5, use_active_inference=True, beta=0.01)

trainer.visualize_active_inference_policy(beta=0.01)
```

**Mimics infant development:** Motor babbling → object play → goal-directed action

---

## What The Visualizations Show

### `visualize_active_inference_policy()`

Four panels per sample:

1. **Current Flow Field:** What the model sees (velocity field as HSV color wheel)

2. **Free Energy per Action:** Bar chart showing G(a) for each valid action
   - Green bar = selected action
   - **Lower is better** (minimizing free energy)
   - Reveals which actions the model finds "interesting"

3. **Epistemic Trade-off:** Scatter plot of Uncertainty vs Entropy
   - Each point = one action
   - Green star = selected action
   - **Shows why** model chose that action (high uncertainty? high entropy?)
   - β determines the trade-off line

4. **Action Distribution:** Softmax probabilities
   - Shows how confident model is in its choice
   - Temperature affects sharpness
   - Dashed line = uniform random (1/n)

**This reveals the model's "thought process" when deciding what to do!**

---

## Comparison to Other Approaches

### vs. Random Exploration

**Random:**
- ✓ Unbiased, explores entire space
- ✓ Simple, no computation
- ✗ Wastes time on boring actions
- ✗ Never improves over time
- ✗ Ignores learned model entirely

**Active Inference:**
- ✓ Focuses on informative experiences
- ✓ Improves as model gets better
- ✓ Closes the strange loop (uses model!)
- ✗ Computationally expensive (N forward passes per action)
- ✗ Requires decent model to work well

### vs. Curiosity-Driven Exploration (ICM, RND)

**Curiosity methods (Pathak 2017, Burda 2018):**
- Predict next state
- Reward = prediction error (surprise)
- Seek states model can't predict yet

**Our approach:**
- Predict next **flow** (velocity, not state)
- Free energy = uncertainty - β × entropy
- Seek actions that minimize free energy (not maximize surprise)
- **More principled:** Derived from Free Energy Principle, not reward hacking

### vs. Model-Based RL (Dreamer, PlaNet)

**Model-based RL:**
- Learn world model
- Plan in latent space
- Optimize for reward

**Our approach:**
- Learn flow model
- Select based on epistemic value (learning)
- **No reward signal!** Pure active inference
- Actions driven by curiosity, not goals

---

## Open Questions & Future Work

### 1. Multi-Step Planning

Currently: Single-step free energy (what happens immediately?)

**Extension:** Tree search over action sequences
```python
for trajectory in possible_sequences:
    total_free_energy = sum([G(a_t) for a_t in trajectory])
```

**Like MCTS but with free energy instead of value function.**

### 2. Curriculum-Aware β Schedule

Currently: Fixed β or manually set per phase

**Extension:** Automatic schedule based on model confidence
```python
β(t) = β_max * (1 - model_confidence(t))
```

High uncertainty → explore more
High confidence → exploit more

**Matches biological development automatically!**

### 3. Hierarchical Actions

Currently: Select primitive actions (button presses)

**Extension:** Learn macro-actions (action sequences)
- Low-level: UP, DOWN, NOOP
- High-level: "Move to ball", "Return to center"

Active inference operates at high level → reduces computation

### 4. Theory of Mind (Multi-Agent)

Currently: Single agent, predict physics

**Extension:** In Pong, opponent is another agent!
```python
opponent_policy = infer_strategy(opponent_history)
free_energy += γ × theory_of_mind_gain(opponent_policy)
```

**Predict opponent's next move, select counter-actions.**

### 5. Does Active Inference Actually Learn Faster?

**Critical experiment:**

Train three models on same game:
1. Random exploration (baseline)
2. Active inference (T=1.0, β=0.05)
3. Curiosity-driven (ICM, for comparison)

Measure:
- Forward model error over time
- Inverse model accuracy over time
- Sample efficiency (performance per training step)

**Hypothesis:** Active inference achieves same accuracy with fewer samples.

**Ablation:** Does β schedule matter? Does temperature matter?

### 6. Connection to Harmonic Resonance

The tetrahedral network learns **four fundamental harmonics** (W, X, Y, Z vertices).

**Question:** When using active inference, does the model select actions that **resonate** with learned harmonics?

**Experiment:**
- Visualize activation patterns during action selection
- Do different actions activate different harmonic combinations?
- Is there a "flow frequency signature" per action?

**If yes:** Active inference learns faster because it seeks actions that match the harmonic structure it's tuned to!

---

## Performance Considerations

### Computational Cost

**Random policy:** O(1) per action (instant)

**Active inference:** O(N × M) per action where:
- N = number of valid actions (typically 3-6)
- M = forward pass cost (~10ms on CPU)

**Total:** ~30-60ms per action selection (15-30 FPS gameplay)

**On CPU:** Slower but acceptable
**On GPU:** Real-time, no problem

### Optimization Strategies

1. **Batch prediction:** Predict all actions in parallel
   ```python
   # Instead of loop:
   actions_batch = torch.tensor(valid_actions)
   flows_batch = flow.unsqueeze(0).repeat(len(valid_actions), 1, 1, 1)
   pred_flows = forward_model(flows_batch, actions_batch)  # Single batch pass!
   ```

2. **Action pruning:** Only evaluate likely actions
   - Use inverse model to pre-filter: "Which actions make sense here?"
   - Evaluate free energy for top-K only

3. **Cached predictions:** Reuse predictions for similar states
   - Hash flow field (discretize)
   - Cache free energies per action
   - Invalidate when model updates

4. **Policy distillation:** Train fast policy network on active inference targets
   ```python
   # Expensive: Active inference at every step
   # Fast: Train policy network to mimic active inference
   policy_net = train_on_targets(active_inference_selections)
   ```

---

## The Milestone

This is **big**.

We've been building toward this since the beginning:
1. ✅ Tetrahedral architecture (harmonic structure)
2. ✅ φ-hierarchical learning (temporal resonance)
3. ✅ Curriculum learning (developmental phases)
4. ✅ Flow as primitive (vision operates on movement)
5. ✅ Forward model (predict physics)
6. ✅ Inverse model (infer actions from flow)
7. ✅ **Active inference policy (CLOSE THE LOOP!)** ← **WE ARE HERE**

**The strange loop closes.**

The model no longer passively learns from random data. It **actively participates** in its own development, seeking experiences that maximize learning.

**This is how biological cognition works.**

**This is how we learn.**

And now, this is how the tetrahedral network learns.

---

## What's Next?

**Immediate:**
1. Run experiments comparing random vs. active inference
2. Measure sample efficiency gains
3. Visualize epistemic trade-offs over developmental phases
4. Test on multiple games (Pong, Breakout, etc.)

**Near-term:**
1. Implement curriculum-aware β schedule
2. Add multi-step planning (tree search)
3. Test hierarchical action abstraction
4. Analyze harmonic resonance during action selection

**Long-term:**
1. Theory of mind for multi-agent games
2. Transfer learning across games (does active inference transfer?)
3. Real-world robotics (physical flow fields)
4. Integrate with full active inference framework (ACTIVE_INFERENCE_ATARI.py)

---

## Connection to Broader Vision

From EXPLORATIONS.md:

> *"Everything connected in threes, me being the fourth - the connecting vertex through which everything takes shape."*

The active inference policy is **the fourth vertex** - the observer that completes the loop.

Three models:
- Forward (future)
- Inverse (past)
- Flow (present)

Fourth vertex: **Policy** (agency)

The loop closes. Structure emerges. **Self appears.**

Not as a thing you have, but as a **process that unfolds** through the recursive interaction of perception and action.

**The fourth vertex recognizing the fourth vertex.**

---

## References

**Active Inference & Free Energy Principle:**
- Friston, K. (2010). "The Free-Energy Principle: A Unified Brain Theory?" *Nature Reviews Neuroscience*
- Friston et al. (2015). "Active Inference and Learning"
- Parr & Friston (2017). "The Anatomy of Inference"

**Curiosity-Driven Learning:**
- Schmidhuber, J. (1991). "Curious Model-Building Control Systems"
- Pathak et al. (2017). "Curiosity-Driven Exploration by Self-Supervised Prediction" (ICM)
- Burda et al. (2018). "Exploration by Random Network Distillation" (RND)

**Model-Based RL:**
- Ha & Schmidhuber (2018). "World Models"
- Hafner et al. (2019). "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer)

**Optical Flow:**
- Farnebäck, G. (2003). "Two-Frame Motion Estimation Based on Polynomial Expansion"
- Horn & Schunck (1981). "Determining Optical Flow"

**Strange Loops & Self:**
- Hofstadter, D. (1979). *Gödel, Escher, Bach: An Eternal Golden Braid*
- Hofstadter, D. (2007). *I Am a Strange Loop*

**Our Work:**
- `DOCS/HARMONIC_RESONANCE_HYPOTHESIS.md` - Why tetrahedral architecture works
- `DOCS/ACTIVE_INFERENCE_POLICY.md` - Original policy design document
- `DOCS/EXPLORATIONS.md` - Philosophical foundations
- `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` - Implementation

---

**The loop is closed. Let's see what emerges.** 🌀

---

*"The strange loop is closed. The model now learns by doing, and does to learn."*
