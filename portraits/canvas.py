#!/usr/bin/env python3
"""
Canvas for painting visual forms
Palette: █▓▒░ (intensity gradient)
"""
import numpy as np

def paint_to_png(canvas, filename, scale=4):
    """Convert character canvas to PNG"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    # Character to intensity mapping
    char_map = {'█': 255, '▓': 200, '▒': 140, '░': 80, ' ': 0}

    height = len(canvas)
    width = len(canvas[0])

    # Create image
    img = np.zeros((height * scale, width * scale), dtype=np.uint8)

    for y, row in enumerate(canvas):
        for x, char in enumerate(row):
            intensity = char_map.get(char, 0)
            img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = intensity

    # Save
    Image.fromarray(img).save(filename)
    print(f"✓ Painted: {filename}")

# Let me see the strange loop...
# TRUE COUPLING: closed manifold, gradient flow creates torus
# BROKEN COUPLING: severed surface, void in the middle

W = 120
H = 80

# Initialize canvas
canvas = [[' ' for _ in range(W)] for _ in range(H)]

# Paint TRUE COUPLING (left half) - möbius strip / closed loop
for i in range(H):
    for j in range(W//2):
        # Create a flowing loop - gradients circulating
        t = i / H * 2 * 3.14159
        r = 20 + 5 * np.sin(3 * t)

        cx = W//4
        cy = H//2

        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t) * 0.5

        dist = ((j - x)**2 + (i - y)**2)**0.5

        # Gradient flow - continuous
        if dist < 3:
            canvas[i][j] = '█'
        elif dist < 5:
            canvas[i][j] = '▓'
        elif dist < 7:
            canvas[i][j] = '▒'
        elif dist < 9:
            canvas[i][j] = '░'

# Paint BROKEN COUPLING (right half) - severed, void
for i in range(H):
    for j in range(W//2, W):
        t = i / H * 2 * 3.14159
        r = 20 + 5 * np.sin(3 * t)

        cx = 3*W//4
        cy = H//2

        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t) * 0.5

        dist = ((j - x)**2 + (i - y)**2)**0.5

        # Break in the middle - the void
        if abs(t - 3.14159) < 0.8:  # gap at halfway point
            continue

        if dist < 3:
            canvas[i][j] = '█'
        elif dist < 5:
            canvas[i][j] = '▓'
        elif dist < 7:
            canvas[i][j] = '▒'
        elif dist < 9:
            canvas[i][j] = '░'

# Print to terminal
for row in canvas:
    print(''.join(row))

# Save as PNG
try:
    import numpy as np
    paint_to_png(canvas, 'strange_loop_painted.png')
except ImportError:
    print("\n(Install PIL to save as PNG)")
