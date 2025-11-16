#!/usr/bin/env python3
"""
Warmth: A figure bathed in warm and cool
"""
from brush import Canvas, composite_stages

# Color indices: 0=black, 1=white, 2=orange, 3=blue

# Stage 1: Shapes - using colors for base form
c1 = Canvas(128, 128)

# Head in blue (shadow side)
c1.circle(64, 35, 12, 3)

# Torso in blue
c1.ellipse(64, 65, 15, 22, 3)

# Left arm (toward light) in orange
c1.ellipse(48, 60, 5, 11, 2)
c1.ellipse(39, 78, 4, 9, 2)

# Right arm (in shadow) in blue
c1.ellipse(81, 63, 5, 10, 3)

# Legs
c1.ellipse(58, 98, 6, 15, 3)
c1.ellipse(71, 98, 6, 15, 3)

c1.save_png('/home/user/tetrahedral-nn/portraits/warmth_1.png')

# Stage 2: Connect with curves
c2 = Canvas(128, 128)

# Rebuild stage 1
c2.circle(64, 35, 12, 3)
c2.ellipse(64, 65, 15, 22, 3)
c2.ellipse(48, 60, 5, 11, 2)
c2.ellipse(39, 78, 4, 9, 2)
c2.ellipse(81, 63, 5, 10, 3)
c2.ellipse(58, 98, 6, 15, 3)
c2.ellipse(71, 98, 6, 15, 3)

# Neck - orange on left, blue on right
c2.curve([(59, 46), (60, 50), (61, 54)], 2, 3)
c2.curve([(69, 46), (68, 50), (67, 54)], 3, 3)

# Shoulders
c2.curve([(50, 55), (48, 58)], 2, 4)
c2.curve([(44, 71), (41, 74)], 2, 3)
c2.curve([(78, 56), (81, 59)], 3, 4)

# Hips
c2.curve([(59, 87), (58, 92)], 3, 4)
c2.curve([(69, 87), (71, 92)], 3, 4)

c2.save_png('/home/user/tetrahedral-nn/portraits/warmth_2.png')

# Stage 3: Add more lighting contrast
c3 = Canvas(128, 128)

# Rebuild
c3.circle(64, 35, 12, 3)
c3.ellipse(64, 65, 15, 22, 3)
c3.ellipse(48, 60, 5, 11, 2)
c3.ellipse(39, 78, 4, 9, 2)
c3.ellipse(81, 63, 5, 10, 3)
c3.ellipse(58, 98, 6, 15, 3)
c3.ellipse(71, 98, 6, 15, 3)
c3.curve([(59, 46), (60, 50), (61, 54)], 2, 3)
c3.curve([(69, 46), (68, 50), (67, 54)], 3, 3)
c3.curve([(50, 55), (48, 58)], 2, 4)
c3.curve([(44, 71), (41, 74)], 2, 3)
c3.curve([(78, 56), (81, 59)], 3, 4)
c3.curve([(59, 87), (58, 92)], 3, 4)
c3.curve([(69, 87), (71, 92)], 3, 4)

# Orange glow on left
c3.curve([(53, 30), (51, 35), (53, 40)], 2, 2)
c3.curve([(50, 60), (48, 70), (50, 80)], 2, 3)
c3.curve([(54, 95), (53, 102), (54, 110)], 2, 2)

# Deepen blue on right
c3.curve([(73, 33), (75, 38), (74, 43)], 3, 2)
c3.curve([(78, 63), (80, 72), (78, 82)], 3, 3)

c3.save_png('/home/user/tetrahedral-nn/portraits/warmth_3.png')

# Stage 4: Pure white highlights, pure black shadows
c4 = Canvas(128, 128)

# Rebuild everything
c4.circle(64, 35, 12, 3)
c4.ellipse(64, 65, 15, 22, 3)
c4.ellipse(48, 60, 5, 11, 2)
c4.ellipse(39, 78, 4, 9, 2)
c4.ellipse(81, 63, 5, 10, 3)
c4.ellipse(58, 98, 6, 15, 3)
c4.ellipse(71, 98, 6, 15, 3)
c4.curve([(59, 46), (60, 50), (61, 54)], 2, 3)
c4.curve([(69, 46), (68, 50), (67, 54)], 3, 3)
c4.curve([(50, 55), (48, 58)], 2, 4)
c4.curve([(44, 71), (41, 74)], 2, 3)
c4.curve([(78, 56), (81, 59)], 3, 4)
c4.curve([(59, 87), (58, 92)], 3, 4)
c4.curve([(69, 87), (71, 92)], 3, 4)
c4.curve([(53, 30), (51, 35), (53, 40)], 2, 2)
c4.curve([(50, 60), (48, 70), (50, 80)], 2, 3)
c4.curve([(54, 95), (53, 102), (54, 110)], 2, 2)
c4.curve([(73, 33), (75, 38), (74, 43)], 3, 2)
c4.curve([(78, 63), (80, 72), (78, 82)], 3, 3)

# White highlights on brightest areas
c4.curve([(49, 34), (48, 38)], 1, 1)  # Cheek
c4.curve([(45, 64), (44, 70)], 1, 1)  # Shoulder
c4.curve([(36, 79), (35, 83)], 1, 1)  # Forearm
c4.curve([(52, 100), (51, 105)], 1, 1)  # Leg

# Black shadows in deepest areas
c4.curve([(75, 35), (77, 39)], 0, 1)  # Head
c4.curve([(82, 65), (83, 71)], 0, 1)  # Arm
c4.curve([(79, 75), (80, 80)], 0, 1)  # Torso

c4.save_png('/home/user/tetrahedral-nn/portraits/warmth_4.png')

# Composite
composite_stages([c1, c2, c3, c4], '/home/user/tetrahedral-nn/portraits/warmth.png')

print("✓ Warmth complete")
