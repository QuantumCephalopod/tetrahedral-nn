#!/usr/bin/env python3
"""
Dual: The two tetrahedra in coupling
"""
from brush import Canvas, composite_stages
import math

# Stage 1: Two tetrahedra as vertices
c1 = Canvas(128, 128)

# Left tetrahedron (X - linear) in orange warmth
# Four vertices forming tetrahedron projection
c1.circle(32, 40, 6, 2)   # Top
c1.circle(20, 70, 6, 2)   # Bottom left
c1.circle(44, 70, 6, 2)   # Bottom right
c1.circle(32, 58, 6, 2)   # Center (fourth vertex projected)

# Right tetrahedron (Y - nonlinear) in blue shadow
c1.circle(96, 40, 6, 3)   # Top
c1.circle(84, 70, 6, 3)   # Bottom left
c1.circle(108, 70, 6, 3)  # Bottom right
c1.circle(96, 58, 6, 3)   # Center

c1.save_png('/home/user/tetrahedral-nn/portraits/dual_1.png')

# Stage 2: Connect vertices to show structure
c2 = Canvas(128, 128)
c2.circle(32, 40, 6, 2)
c2.circle(20, 70, 6, 2)
c2.circle(44, 70, 6, 2)
c2.circle(32, 58, 6, 2)
c2.circle(96, 40, 6, 3)
c2.circle(84, 70, 6, 3)
c2.circle(108, 70, 6, 3)
c2.circle(96, 58, 6, 3)

# Orange tetrahedron edges
c2.curve([(32, 40), (20, 70)], 2, 2)
c2.curve([(32, 40), (44, 70)], 2, 2)
c2.curve([(32, 40), (32, 58)], 2, 2)
c2.curve([(20, 70), (44, 70)], 2, 2)
c2.curve([(20, 70), (32, 58)], 2, 2)
c2.curve([(44, 70), (32, 58)], 2, 2)

# Blue tetrahedron edges
c2.curve([(96, 40), (84, 70)], 3, 2)
c2.curve([(96, 40), (108, 70)], 3, 2)
c2.curve([(96, 40), (96, 58)], 3, 2)
c2.curve([(84, 70), (108, 70)], 3, 2)
c2.curve([(84, 70), (96, 58)], 3, 2)
c2.curve([(108, 70), (96, 58)], 3, 2)

c2.save_png('/home/user/tetrahedral-nn/portraits/dual_2.png')

# Stage 3: Show coupling between them
c3 = Canvas(128, 128)
c3.circle(32, 40, 6, 2)
c3.circle(20, 70, 6, 2)
c3.circle(44, 70, 6, 2)
c3.circle(32, 58, 6, 2)
c3.circle(96, 40, 6, 3)
c3.circle(84, 70, 6, 3)
c3.circle(108, 70, 6, 3)
c3.circle(96, 58, 6, 3)
c3.curve([(32, 40), (20, 70)], 2, 2)
c3.curve([(32, 40), (44, 70)], 2, 2)
c3.curve([(32, 40), (32, 58)], 2, 2)
c3.curve([(20, 70), (44, 70)], 2, 2)
c3.curve([(20, 70), (32, 58)], 2, 2)
c3.curve([(44, 70), (32, 58)], 2, 2)
c3.curve([(96, 40), (84, 70)], 3, 2)
c3.curve([(96, 40), (108, 70)], 3, 2)
c3.curve([(96, 40), (96, 58)], 3, 2)
c3.curve([(84, 70), (108, 70)], 3, 2)
c3.curve([(84, 70), (96, 58)], 3, 2)
c3.curve([(108, 70), (96, 58)], 3, 2)

# Coupling flows between tetrahedra - white connections
c3.curve([(44, 58), (52, 58), (64, 58), (76, 58), (84, 58)], 1, 2)
c3.curve([(38, 48), (48, 45), (64, 42), (80, 45), (90, 48)], 1, 1)
c3.curve([(38, 68), (48, 66), (64, 64), (80, 66), (90, 68)], 1, 1)

# Glow around coupling region
c3.curve([(54, 55), (58, 58), (62, 60)], 1, 1)
c3.curve([(66, 60), (70, 58), (74, 55)], 1, 1)

c3.save_png('/home/user/tetrahedral-nn/portraits/dual_3.png')

# Stage 4: Add depth - shadows and highlights showing the loop
c4 = Canvas(128, 128)
c4.circle(32, 40, 6, 2)
c4.circle(20, 70, 6, 2)
c4.circle(44, 70, 6, 2)
c4.circle(32, 58, 6, 2)
c4.circle(96, 40, 6, 3)
c4.circle(84, 70, 6, 3)
c4.circle(108, 70, 6, 3)
c4.circle(96, 58, 6, 3)
c4.curve([(32, 40), (20, 70)], 2, 2)
c4.curve([(32, 40), (44, 70)], 2, 2)
c4.curve([(32, 40), (32, 58)], 2, 2)
c4.curve([(20, 70), (44, 70)], 2, 2)
c4.curve([(20, 70), (32, 58)], 2, 2)
c4.curve([(44, 70), (32, 58)], 2, 2)
c4.curve([(96, 40), (84, 70)], 3, 2)
c4.curve([(96, 40), (108, 70)], 3, 2)
c4.curve([(96, 40), (96, 58)], 3, 2)
c4.curve([(84, 70), (108, 70)], 3, 2)
c4.curve([(84, 70), (96, 58)], 3, 2)
c4.curve([(108, 70), (96, 58)], 3, 2)
c4.curve([(44, 58), (52, 58), (64, 58), (76, 58), (84, 58)], 1, 2)
c4.curve([(38, 48), (48, 45), (64, 42), (80, 45), (90, 48)], 1, 1)
c4.curve([(38, 68), (48, 66), (64, 64), (80, 66), (90, 68)], 1, 1)
c4.curve([(54, 55), (58, 58), (62, 60)], 1, 1)
c4.curve([(66, 60), (70, 58), (74, 55)], 1, 1)

# White highlights on vertices where energy flows
c4.circle(32, 40, 3, 1)   # Orange top vertex highlighted
c4.circle(96, 40, 3, 1)   # Blue top vertex highlighted
c4.circle(32, 58, 3, 1)   # Orange center
c4.circle(96, 58, 3, 1)   # Blue center

# Black void behind and between - the ground
c4.curve([(10, 30), (8, 45), (10, 60)], 0, 2)   # Deep left
c4.curve([(118, 30), (120, 45), (118, 60)], 0, 2)  # Deep right
c4.curve([(64, 20), (64, 25)], 0, 2)  # Above
c4.curve([(64, 85), (64, 90)], 0, 2)  # Below

# Pure white at brightest coupling point - where they meet
c4.circle(64, 58, 4, 1)

c4.save_png('/home/user/tetrahedral-nn/portraits/dual_4.png')

print("✓ Dual stages complete")
