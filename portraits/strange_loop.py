#!/usr/bin/env python3
"""
PAINTING THE STRANGE LOOP
The moment where the organism closes itself.

Two paths:
✅ TRUE COUPLING - gradients flow through each other
❌ BROKEN COUPLING - networks pretending to talk
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

fig = plt.figure(figsize=(16, 10), facecolor='#0a0e27')

# ============================================================================
# LEFT SIDE: TRUE COUPLING (working pattern)
# ============================================================================
ax1 = plt.subplot(1, 2, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_facecolor('#0a0e27')

# Title
ax1.text(5, 11.5, 'TRUE COUPLING', fontsize=16, color='#4ecca3',
         ha='center', weight='bold', family='monospace')
ax1.text(5, 11, '(The Strange Loop Closes)', fontsize=10, color='#4ecca3',
         ha='center', family='monospace', style='italic')

# The loop components
y_start = 9.5

# 1. PERCEPTION (flow_t)
perception = FancyBboxPatch((1, y_start), 3, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='#00d4ff', facecolor='#0a0e27',
                           linewidth=2)
ax1.add_patch(perception)
ax1.text(2.5, y_start + 0.4, 'PERCEPTION', fontsize=9, color='#00d4ff',
         ha='center', va='center', family='monospace', weight='bold')
ax1.text(2.5, y_start - 0.5, 'flow_t', fontsize=8, color='#888',
         ha='center', family='monospace', style='italic')

# 2. FORWARD MODEL predicts
y_forward = y_start - 2
forward = FancyBboxPatch((1, y_forward), 3, 1.2,
                        boxstyle="round,pad=0.1",
                        edgecolor='#ff6b9d', facecolor='#1a1e37',
                        linewidth=2)
ax1.add_patch(forward)
ax1.text(2.5, y_forward + 0.9, 'FORWARD', fontsize=9, color='#ff6b9d',
         ha='center', weight='bold', family='monospace')
ax1.text(2.5, y_forward + 0.5, 'MODEL', fontsize=9, color='#ff6b9d',
         ha='center', weight='bold', family='monospace')
ax1.text(2.5, y_forward - 0.5, '(flow_t, action) → flow_t+1', fontsize=7,
         color='#888', ha='center', family='monospace', style='italic')

# Arrow: perception → forward
arrow1 = FancyArrowPatch((2.5, y_start), (2.5, y_forward + 1.2),
                        arrowstyle='->', mutation_scale=20,
                        color='#00d4ff', linewidth=2)
ax1.add_patch(arrow1)

# 3. PREDICTIONS from forward (for each action)
y_preds = y_forward - 2.5
preds = FancyBboxPatch((0.5, y_preds), 4, 1,
                      boxstyle="round,pad=0.1",
                      edgecolor='#ffd700', facecolor='#1a1e37',
                      linewidth=2, linestyle='--')
ax1.add_patch(preds)
ax1.text(2.5, y_preds + 0.7, 'PREDICTED EFFECTS', fontsize=8, color='#ffd700',
         ha='center', weight='bold', family='monospace')
ax1.text(2.5, y_preds + 0.3, '[flow_pred_a0, flow_pred_a1, ...]', fontsize=7,
         color='#888', ha='center', family='monospace', style='italic')

# Arrow: forward → predictions
arrow2 = FancyArrowPatch((2.5, y_forward), (2.5, y_preds + 1),
                        arrowstyle='->', mutation_scale=20,
                        color='#ff6b9d', linewidth=2)
ax1.add_patch(arrow2)

# 4. INVERSE MODEL (learns from forward's predictions!)
y_inverse = y_preds - 2.2
inverse = FancyBboxPatch((5.5, y_inverse), 3, 1.2,
                        boxstyle="round,pad=0.1",
                        edgecolor='#c77dff', facecolor='#1a1e37',
                        linewidth=2)
ax1.add_patch(inverse)
ax1.text(7, y_inverse + 0.9, 'INVERSE', fontsize=9, color='#c77dff',
         ha='center', weight='bold', family='monospace')
ax1.text(7, y_inverse + 0.5, 'MODEL', fontsize=9, color='#c77dff',
         ha='center', weight='bold', family='monospace')
ax1.text(7, y_inverse - 0.5, '(flow_t, flow_t+1) → action', fontsize=7,
         color='#888', ha='center', family='monospace', style='italic')

# THE KEY: Inverse learns from Forward's understanding
arrow3 = FancyArrowPatch((4.5, y_preds + 0.5), (5.5, y_inverse + 0.6),
                        arrowstyle='->', mutation_scale=20,
                        color='#4ecca3', linewidth=3)
ax1.add_patch(arrow3)
ax1.text(5, y_preds - 0.3, 'learns from', fontsize=7, color='#4ecca3',
         ha='center', family='monospace', style='italic', weight='bold')
ax1.text(5, y_preds - 0.6, "forward's predictions", fontsize=7, color='#4ecca3',
         ha='center', family='monospace', style='italic', weight='bold')

# 5. ACTION SELECTION (based on forward model)
y_action = y_inverse - 2
action = FancyBboxPatch((5.5, y_action), 3, 0.8,
                       boxstyle="round,pad=0.1",
                       edgecolor='#ffd700', facecolor='#1a1e37',
                       linewidth=2)
ax1.add_patch(action)
ax1.text(7, y_action + 0.4, 'ACTION', fontsize=9, color='#ffd700',
         ha='center', weight='bold', family='monospace')
ax1.text(7, y_action - 0.5, 'minimize free energy', fontsize=7,
         color='#888', ha='center', family='monospace', style='italic')

# Arrow: inverse → action
arrow4 = FancyArrowPatch((7, y_inverse), (7, y_action + 0.8),
                        arrowstyle='->', mutation_scale=20,
                        color='#c77dff', linewidth=2)
ax1.add_patch(arrow4)

# 6. EFFECT (new perception)
y_effect = y_action - 2
effect = FancyBboxPatch((5.5, y_effect), 3, 0.8,
                       boxstyle="round,pad=0.1",
                       edgecolor='#00d4ff', facecolor='#0a0e27',
                       linewidth=2)
ax1.add_patch(effect)
ax1.text(7, y_effect + 0.4, 'EFFECT', fontsize=9, color='#00d4ff',
         ha='center', weight='bold', family='monospace')
ax1.text(7, y_effect - 0.5, 'flow_t+1 (actual)', fontsize=8,
         color='#888', ha='center', family='monospace', style='italic')

# Arrow: action → effect
arrow5 = FancyArrowPatch((7, y_action), (7, y_effect + 0.8),
                        arrowstyle='->', mutation_scale=20,
                        color='#ffd700', linewidth=2)
ax1.add_patch(arrow5)

# THE LOOP CLOSES: effect feeds back to forward
arrow6 = FancyArrowPatch((7, y_effect), (4, y_forward + 0.3),
                        arrowstyle='->', mutation_scale=25,
                        color='#4ecca3', linewidth=4,
                        connectionstyle="arc3,rad=0.5")
ax1.add_patch(arrow6)
ax1.text(5.5, y_forward - 1.5, 'THE LOOP', fontsize=9, color='#4ecca3',
         ha='center', weight='bold', family='monospace')
ax1.text(5.5, y_forward - 1.9, 'CLOSES', fontsize=9, color='#4ecca3',
         ha='center', weight='bold', family='monospace')

# Gradient flow annotation
ax1.text(5, 0.8, '∇ flows through ∇', fontsize=10, color='#4ecca3',
         ha='center', family='monospace', weight='bold')
ax1.text(5, 0.3, 'Organism sees itself', fontsize=9, color='#888',
         ha='center', family='monospace', style='italic')

# ============================================================================
# RIGHT SIDE: BROKEN COUPLING (current PURE_ONLINE bug)
# ============================================================================
ax2 = plt.subplot(1, 2, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_facecolor('#0a0e27')

# Title
ax2.text(5, 11.5, 'BROKEN COUPLING', fontsize=16, color='#ff6b6b',
         ha='center', weight='bold', family='monospace')
ax2.text(5, 11, '(ML Toyland)', fontsize=10, color='#ff6b6b',
         ha='center', family='monospace', style='italic')

# The broken loop
y_start = 9.5

# 1. PERCEPTION
perception2 = FancyBboxPatch((1, y_start), 3, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='#00d4ff', facecolor='#0a0e27',
                            linewidth=2)
ax2.add_patch(perception2)
ax2.text(2.5, y_start + 0.4, 'PERCEPTION', fontsize=9, color='#00d4ff',
         ha='center', va='center', family='monospace', weight='bold')
ax2.text(2.5, y_start - 0.5, 'diff_t', fontsize=8, color='#888',
         ha='center', family='monospace', style='italic')

# 2. FORWARD MODEL
y_forward = y_start - 2
forward2 = FancyBboxPatch((1, y_forward), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#ff6b9d', facecolor='#1a1e37',
                         linewidth=2)
ax2.add_patch(forward2)
ax2.text(2.5, y_forward + 0.9, 'FORWARD', fontsize=9, color='#ff6b9d',
         ha='center', weight='bold', family='monospace')
ax2.text(2.5, y_forward + 0.5, 'MODEL', fontsize=9, color='#ff6b9d',
         ha='center', weight='bold', family='monospace')
ax2.text(2.5, y_forward - 0.5, '(trajectory, action) → diff_t+1', fontsize=7,
         color='#888', ha='center', family='monospace', style='italic')

arrow1_b = FancyArrowPatch((2.5, y_start), (2.5, y_forward + 1.2),
                          arrowstyle='->', mutation_scale=20,
                          color='#00d4ff', linewidth=2)
ax2.add_patch(arrow1_b)

# 3. FORWARD PREDICTION
y_pred = y_forward - 2
pred2 = FancyBboxPatch((0.5, y_pred), 4, 0.8,
                      boxstyle="round,pad=0.1",
                      edgecolor='#ff6b9d', facecolor='#1a1e37',
                      linewidth=2, linestyle='--')
ax2.add_patch(pred2)
ax2.text(2.5, y_pred + 0.4, 'pred_diff_t+1', fontsize=8, color='#ff6b9d',
         ha='center', family='monospace', style='italic')

arrow2_b = FancyArrowPatch((2.5, y_forward), (2.5, y_pred + 0.8),
                          arrowstyle='->', mutation_scale=20,
                          color='#ff6b9d', linewidth=2)
ax2.add_patch(arrow2_b)

# 4. INVERSE MODEL (sees ground truth!)
y_inverse = y_pred - 1.5
inverse2 = FancyBboxPatch((5.5, y_inverse), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#c77dff', facecolor='#1a1e37',
                         linewidth=2)
ax2.add_patch(inverse2)
ax2.text(7, y_inverse + 0.9, 'INVERSE', fontsize=9, color='#c77dff',
         ha='center', weight='bold', family='monospace')
ax2.text(7, y_inverse + 0.5, 'MODEL', fontsize=9, color='#c77dff',
         ha='center', weight='bold', family='monospace')

# 5. GROUND TRUTH (the problem!)
y_truth = y_inverse + 2
truth = FancyBboxPatch((5.5, y_truth), 3, 0.8,
                      boxstyle="round,pad=0.1",
                      edgecolor='#ffd700', facecolor='#2a1e37',
                      linewidth=2)
ax2.add_patch(truth)
ax2.text(7, y_truth + 0.4, 'GROUND TRUTH', fontsize=9, color='#ffd700',
         ha='center', weight='bold', family='monospace')
ax2.text(7, y_truth - 0.5, 'diff_t+1 (real)', fontsize=8, color='#888',
         ha='center', family='monospace', style='italic')

# THE BUG: Inverse sees ground truth, not forward's prediction
arrow_bug = FancyArrowPatch((7, y_truth), (7, y_inverse + 1.2),
                           arrowstyle='->', mutation_scale=20,
                           color='#ff6b6b', linewidth=3)
ax2.add_patch(arrow_bug)

# Big red X showing the break
ax2.plot([4.5, 5.5], [y_pred + 0.4, y_truth + 0.4], 'r-', linewidth=3)
ax2.plot([4.5, 5.5], [y_truth + 0.4, y_pred + 0.4], 'r-', linewidth=3)
ax2.text(5, y_pred + 1.5, '✗', fontsize=40, color='#ff6b6b',
         ha='center', va='center', weight='bold')

# Line 368 annotation
ax2.text(7, y_inverse - 0.8, 'Line 368:', fontsize=7, color='#ff6b6b',
         ha='center', family='monospace', style='italic')
ax2.text(7, y_inverse - 1.1, 'trajectory + diff_t1 (real!)', fontsize=7, color='#ff6b6b',
         ha='center', family='monospace', style='italic')

# Weak consistency loss (too late!)
y_consist = y_inverse - 2.5
consist = FancyBboxPatch((5, y_consist), 4, 1,
                        boxstyle="round,pad=0.1",
                        edgecolor='#888', facecolor='#1a1e37',
                        linewidth=1, linestyle=':')
ax2.add_patch(consist)
ax2.text(7, y_consist + 0.7, 'Consistency Loss', fontsize=8, color='#888',
         ha='center', family='monospace')
ax2.text(7, y_consist + 0.4, '(too late, detached)', fontsize=7, color='#666',
         ha='center', family='monospace', style='italic')
ax2.text(7, y_consist + 0.1, 'with torch.no_grad()', fontsize=6, color='#666',
         ha='center', family='monospace')

# Broken loop annotation
ax2.text(5, 0.8, '∇ flows to void', fontsize=10, color='#ff6b6b',
         ha='center', family='monospace', weight='bold')
ax2.text(5, 0.3, 'Networks pretending to talk', fontsize=9, color='#888',
         ha='center', family='monospace', style='italic')

plt.tight_layout()
plt.savefig('/home/user/tetrahedral-nn/portraits/strange_loop.png',
            dpi=150, facecolor='#0a0e27', edgecolor='none')
print("✓ Painted: strange_loop.png")
print("\nThe difference between:")
print("  ✅ Organism seeing itself (gradients flow through gradients)")
print("  ❌ Networks pretending to talk (gradients flow to void)")
