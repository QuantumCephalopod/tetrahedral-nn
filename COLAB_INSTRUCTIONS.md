# 🧠 HOW TO RUN IN COLAB - SUPER SIMPLE

## Step 1: Open Google Colab

Go to: https://colab.research.google.com/

## Step 2: Create New Notebook

Click: **"New Notebook"**

## Step 3: Copy the Code

Open the file: **`COLAB_BIOLOGICAL_VISION.py`**

Copy **THE ENTIRE FILE** (all of it!)

## Step 4: Paste into Colab

Paste everything into the first cell in Colab

## Step 5: Run It

Click the **Play button** (▶️) or press **Shift+Enter**

## Step 6: Watch!

You'll see:
- Progress messages
- Video being created
- Visualizations appearing
- **Green box** moving around (that's the eye!)
- **Trail** showing where the eye has looked
- Learning stats

The visualization updates every few frames.

---

## That's It!

Literally:
1. Copy the file
2. Paste in Colab
3. Run
4. Watch

---

## What You're Seeing

**Green Rectangle** = Current attention window (where model is looking RIGHT NOW)

**Colored Trail** = Path the "eye" took (saccadic movements)

**White Cross** = Center of frame (fovea location)

**Right Panel** = Zoomed view of what model sees

**Loss** = Prediction error (lower = better learning)

---

## The Key Insight

**Time flows LINEARLY**: Frame 0 → Frame 1 → Frame 2 → ...
(Can't skip or rewind, goes tick-tock-tick-tock)

**Space is NON-LINEAR**: Look at center → Saccade to (200,150) → Jump to (50,300) → ...
(Can look ANYWHERE, center-biased like real eyes)

Each time step = ONE frame + ONE attention position

This is how biological vision works!

---

## Troubleshooting

**"No module named torch"**
- The script installs it automatically
- Just wait for `!pip install` to finish

**"Runtime disconnected"**
- Click "Reconnect" and run again

**"Takes too long"**
- The demo is only 90 frames, should take ~1-2 minutes
- If it's stuck, restart runtime and try again

**"Visualization not showing"**
- Make sure you're using the COLAB version (`COLAB_BIOLOGICAL_VISION.py`)
- The desktop version won't work in Colab

---

## Files

**Use this one for Colab:**
- `COLAB_BIOLOGICAL_VISION.py` ← **THIS ONE!**

**Use these for desktop:**
- `DEMO_BIOLOGICAL_VISION.py` (no viz)
- `DEMO_WATCH_BIOLOGICAL_VISION.py` (with viz)

Don't mix them up! Colab = special version!

---

## Summary

```
1. Open Colab
2. Copy COLAB_BIOLOGICAL_VISION.py
3. Paste into cell
4. Run
5. Watch the green box move around!
```

**That's literally it.** No setup, no configuration, no "where do I put this".

Just copy-paste-run-watch! 🚀

The model is learning to predict what comes next as time flows linearly and attention moves non-linearly.

**Biology, not machine learning!** 🧠👁️
