# HOW TO RUN BIOLOGICAL VISION - Simple Instructions

## What This Does

Shows you a **biological vision system** that:
- Time flows **linearly** (tick → tock, frame by frame)
- Space is **non-linear** (eye looks around, center-biased)
- Like a real eye watching the world!

---

## Option 1: Just Run It (No Visualization)

**What to do:**
```bash
python DEMO_BIOLOGICAL_VISION.py
```

**What happens:**
- Creates a test video (bouncing shapes)
- Model watches it with biological vision
- Shows learning progress in terminal
- Saves trained model

**That's it!** Just run that one command.

---

## Option 2: Watch With Visualization (COOL!)

**What to do:**
```bash
python DEMO_WATCH_BIOLOGICAL_VISION.py
```

**What you'll see:**
- Live video with attention window highlighted
- Eye movement trail (saccadic path)
- What the model is looking at (zoomed view)
- Real-time learning stats

**Controls:**
- Press `q` to quit

**This is the COOL one** - you actually SEE the biological vision!

---

## What Each File Does

### Files You RUN:
- `DEMO_BIOLOGICAL_VISION.py` → Run learning, no visualization
- `DEMO_WATCH_BIOLOGICAL_VISION.py` → Run with COOL visualization

### Files That Make It Work:
- `BIOLOGICAL_VISION_SYSTEM.py` → The actual biological vision code
- `TIME_SPACE_ARCHITECTURE.md` → Explains why it works this way
- `CONTINUOUS_LEARNING_SYSTEM.py` → Learning engine
- `Z_interface_coupling.py` → Neural network model

### Generated Files:
- `test_video.mp4` → Test video (auto-created)
- `biological_vision_checkpoint.pt` → Saved model

---

## Understanding What You See

### Terminal Output
```
t=100 | Frame 100 | Pos (245, 180) | Loss: 0.234567 | FPS: 45.2 | Spatial diversity: 87.3
```

**What this means:**
- `t=100` → 100 time steps (100 frames seen)
- `Frame 100` → Currently on video frame 100
- `Pos (245, 180)` → Eye is looking at position (245, 180) in the frame
- `Loss: 0.234567` → Prediction error (lower = better)
- `FPS: 45.2` → Processing 45 frames per second
- `Spatial diversity: 87.3` → How much the eye is moving around

### Visualization Window
- **Green rectangle** → Current attention window
- **Colored trail** → Where eye has been looking (saccadic path)
- **White crosshair** → Center of frame (fovea location)
- **Right panel** → Zoomed view of what model sees

---

## The Key Insight

**Old way (wrong):**
```
Scan all positions in Frame 1 → Then scan all positions in Frame 2
```
Treats time and space the same.

**New way (biological):**
```
Frame 1 + Look at center → Frame 2 + Look somewhere → Frame 3 + Look elsewhere
```
Time is linear. Space is non-linear. Like a real eye!

---

## Troubleshooting

**If you get "No module named 'torch'":**
- Need to install PyTorch first
- Run: `pip install torch torchvision opencv-python numpy`

**If visualization doesn't show:**
- Make sure you have a display (GUI environment)
- Or just use Option 1 (no visualization)

**If it's slow:**
- Reduce max_frames in the code
- Or use smaller window_size

---

## Quick Summary

**To see it work:**
1. Run: `python DEMO_WATCH_BIOLOGICAL_VISION.py`
2. Watch the green box move around
3. See the eye trail showing saccadic movements
4. Press `q` when done

**That's literally it!** No configuration, no setup, just run and watch.

The model is learning to predict what it sees next as time flows linearly and its attention moves non-linearly through space.

**This is biology, not machine learning!** 🧠👁️
