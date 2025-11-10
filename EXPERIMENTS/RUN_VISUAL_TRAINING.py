"""
ONE-CLICK VISUAL TRAINING
=========================

Just run this cell - it does EVERYTHING with visualization!

No more blind flying!
"""

# Import everything needed
import sys
sys.path.append('/home/user/tetrahedral-nn')

# Load architecture
exec(open('/home/user/tetrahedral-nn/W_FOUNDATION/W_geometry.py').read())
exec(open('/home/user/tetrahedral-nn/X_LINEAR/X_linear_tetrahedron.py').read())
exec(open('/home/user/tetrahedral-nn/Y_NONLINEAR/Y_nonlinear_tetrahedron.py').read())
exec(open('/home/user/tetrahedral-nn/Z_COUPLING/Z_interface_coupling.py').read())

# Load fixed trainer
from MOTION_INVERSE_FIXED import FixedTrainer
from VISUAL_TRAINING import train_with_live_visualization

print("\n" + "="*70)
print("🎬 STARTING VISUAL TRAINING")
print("="*70)
print("\nThis will show you:")
print("  • Training curves (forward, inverse, accuracy)")
print("  • Confusion matrix (is it learning causality?)")
print("  • Example predictions with images")
print("  • All updating live every 2 episodes!")
print("\nLook for:")
print("  ✅ Accuracy > 20% (beating random)")
print("  ✅ Inverse loss < 1.75 (learning)")
print("  ✅ Diagonal in confusion matrix (action understanding)")
print("="*70 + "\n")

# Create trainer
trainer = FixedTrainer(
    env_name='ALE/Pong-v5',
    state_dim=128,
    latent_dim=64,  # Smaller for faster training
    base_lr=0.001,   # Higher LR
    batch_size=32    # Bigger batches
)

# Train with live visualization!
train_with_live_visualization(
    trainer,
    n_episodes=10,
    steps_per_episode=50,
    update_every=2  # Update plots every 2 episodes
)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# Final summary
if len(trainer.history['accuracy']) > 0:
    final_acc = trainer.history['accuracy'][-1] * 100
    final_inv = trainer.history['inverse'][-1]

    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {final_acc:.2f}% (random = 16.67%)")
    print(f"   Inverse loss: {final_inv:.4f} (random = {1.791:.4f})")

    if final_acc > 25:
        print("\n✅ SUCCESS! Model learned action causality!")
    elif final_acc > 20:
        print("\n🟡 PARTIAL SUCCESS - Learning but needs more training")
    else:
        print("\n⚠️  NOT LEARNING - Still near random baseline")

print("\n" + "="*70 + "\n")
