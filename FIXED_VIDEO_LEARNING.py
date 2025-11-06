"""
FIXED: Continuous Learning - Proper video sizes
"""

# Quick fix for the size issue:

# OPTION 1: Try 128x128 (better detail, still reasonable)
learner_128 = create_continuous_learner(
    img_size=128,  # 4x more pixels than 64x64
    window_size=3,
    learning_rate=0.0001,
    device='cpu'  # Change to 'cuda' if you have GPU
)

learner_128.watch_video(
    video_path="/content/drive/MyDrive/YouCut_20250910_233053346.mp4",
    target_size=(128, 128),  # Must match img_size!
    report_every=100
)

# OPTION 2: Try 256x256 (much better detail, if you have the compute)
# learner_256 = create_continuous_learner(
#     img_size=256,
#     window_size=3,
#     learning_rate=0.0001,
#     device='cuda'  # Really need GPU for this
# )
#
# learner_256.watch_video(
#     video_path="/content/drive/MyDrive/YouCut_20250910_233053346.mp4",
#     target_size=(256, 256),
#     report_every=50
# )

print("✓ Fixed! Now watching with proper size...")
