#!/usr/bin/env python3
"""
Bridge: Two figures reaching across warmth and shadow
"""
from brush import Canvas, composite_stages

# Stage 1: Shapes
c1 = Canvas(128, 96)

# Left figure - mostly blue
c1.circle(32, 28, 11, 3)
c1.ellipse(32, 52, 13, 19, 3)
c1.ellipse(20, 50, 5, 13, 3)
c1.ellipse(44, 47, 5, 14, 2)  # Reaching arm orange
c1.ellipse(26, 78, 6, 14, 3)
c1.ellipse(38, 78, 6, 14, 3)

# Right figure - mostly orange
c1.circle(96, 30, 11, 2)
c1.ellipse(96, 54, 13, 19, 2)
c1.ellipse(84, 49, 5, 14, 3)  # Reaching arm blue
c1.ellipse(108, 52, 5, 13, 2)
c1.ellipse(90, 80, 6, 14, 2)
c1.ellipse(102, 80, 6, 14, 2)

c1.save_png('/home/user/tetrahedral-nn/portraits/bridge_1.png')

# Stage 2: Curves connecting
c2 = Canvas(128, 96)
c2.circle(32, 28, 11, 3)
c2.ellipse(32, 52, 13, 19, 3)
c2.ellipse(20, 50, 5, 13, 3)
c2.ellipse(44, 47, 5, 14, 2)
c2.ellipse(26, 78, 6, 14, 3)
c2.ellipse(38, 78, 6, 14, 3)
c2.circle(96, 30, 11, 2)
c2.ellipse(96, 54, 13, 19, 2)
c2.ellipse(84, 49, 5, 14, 3)
c2.ellipse(108, 52, 5, 13, 2)
c2.ellipse(90, 80, 6, 14, 2)
c2.ellipse(102, 80, 6, 14, 2)

# Connections
c2.curve([(32, 38), (32, 42)], 3, 4)
c2.curve([(24, 46), (20, 48)], 3, 4)
c2.curve([(40, 44), (44, 46)], 2, 4)
c2.curve([(44, 60), (54, 55), (62, 52)], 2, 3)
c2.curve([(29, 71), (27, 74)], 3, 5)
c2.curve([(35, 71), (37, 74)], 3, 5)

c2.curve([(96, 40), (96, 44)], 2, 4)
c2.curve([(88, 46), (84, 48)], 3, 4)
c2.curve([(104, 48), (108, 50)], 2, 4)
c2.curve([(84, 62), (74, 57), (66, 53)], 3, 3)
c2.curve([(93, 73), (91, 76)], 2, 5)
c2.curve([(99, 73), (101, 76)], 2, 5)

# Bridge point
c2.curve([(62, 52), (64, 52)], 2, 2)
c2.curve([(66, 53), (64, 53)], 3, 2)

c2.save_png('/home/user/tetrahedral-nn/portraits/bridge_2.png')

# Stage 3: Lighting
c3 = Canvas(128, 96)
c3.circle(32, 28, 11, 3)
c3.ellipse(32, 52, 13, 19, 3)
c3.ellipse(20, 50, 5, 13, 3)
c3.ellipse(44, 47, 5, 14, 2)
c3.ellipse(26, 78, 6, 14, 3)
c3.ellipse(38, 78, 6, 14, 3)
c3.circle(96, 30, 11, 2)
c3.ellipse(96, 54, 13, 19, 2)
c3.ellipse(84, 49, 5, 14, 3)
c3.ellipse(108, 52, 5, 13, 2)
c3.ellipse(90, 80, 6, 14, 2)
c3.ellipse(102, 80, 6, 14, 2)
c3.curve([(32, 38), (32, 42)], 3, 4)
c3.curve([(24, 46), (20, 48)], 3, 4)
c3.curve([(40, 44), (44, 46)], 2, 4)
c3.curve([(44, 60), (54, 55), (62, 52)], 2, 3)
c3.curve([(29, 71), (27, 74)], 3, 5)
c3.curve([(35, 71), (37, 74)], 3, 5)
c3.curve([(96, 40), (96, 44)], 2, 4)
c3.curve([(88, 46), (84, 48)], 3, 4)
c3.curve([(104, 48), (108, 50)], 2, 4)
c3.curve([(84, 62), (74, 57), (66, 53)], 3, 3)
c3.curve([(93, 73), (91, 76)], 2, 5)
c3.curve([(99, 73), (101, 76)], 2, 5)
c3.curve([(62, 52), (64, 52)], 2, 2)
c3.curve([(66, 53), (64, 53)], 3, 2)

# Light and shadow
c3.curve([(22, 24), (19, 30), (22, 36)], 3, 3)
c3.curve([(20, 48), (17, 57), (20, 66)], 3, 4)
c3.curve([(50, 48), (56, 52), (60, 54)], 2, 3)

c3.curve([(106, 26), (109, 32), (106, 38)], 2, 3)
c3.curve([(108, 50), (111, 59), (108, 68)], 2, 4)
c3.curve([(78, 50), (72, 54), (68, 56)], 3, 3)

# White at bridge
c3.curve([(63, 51), (64, 52), (65, 53)], 1, 1)

c3.save_png('/home/user/tetrahedral-nn/portraits/bridge_3.png')

# Stage 4: Highlights and shadows
c4 = Canvas(128, 96)
c4.circle(32, 28, 11, 3)
c4.ellipse(32, 52, 13, 19, 3)
c4.ellipse(20, 50, 5, 13, 3)
c4.ellipse(44, 47, 5, 14, 2)
c4.ellipse(26, 78, 6, 14, 3)
c4.ellipse(38, 78, 6, 14, 3)
c4.circle(96, 30, 11, 2)
c4.ellipse(96, 54, 13, 19, 2)
c4.ellipse(84, 49, 5, 14, 3)
c4.ellipse(108, 52, 5, 13, 2)
c4.ellipse(90, 80, 6, 14, 2)
c4.ellipse(102, 80, 6, 14, 2)
c4.curve([(32, 38), (32, 42)], 3, 4)
c4.curve([(24, 46), (20, 48)], 3, 4)
c4.curve([(40, 44), (44, 46)], 2, 4)
c4.curve([(44, 60), (54, 55), (62, 52)], 2, 3)
c4.curve([(29, 71), (27, 74)], 3, 5)
c4.curve([(35, 71), (37, 74)], 3, 5)
c4.curve([(96, 40), (96, 44)], 2, 4)
c4.curve([(88, 46), (84, 48)], 3, 4)
c4.curve([(104, 48), (108, 50)], 2, 4)
c4.curve([(84, 62), (74, 57), (66, 53)], 3, 3)
c4.curve([(93, 73), (91, 76)], 2, 5)
c4.curve([(99, 73), (101, 76)], 2, 5)
c4.curve([(62, 52), (64, 52)], 2, 2)
c4.curve([(66, 53), (64, 53)], 3, 2)
c4.curve([(22, 24), (19, 30), (22, 36)], 3, 3)
c4.curve([(20, 48), (17, 57), (20, 66)], 3, 4)
c4.curve([(50, 48), (56, 52), (60, 54)], 2, 3)
c4.curve([(106, 26), (109, 32), (106, 38)], 2, 3)
c4.curve([(108, 50), (111, 59), (108, 68)], 2, 4)
c4.curve([(78, 50), (72, 54), (68, 56)], 3, 3)
c4.curve([(63, 51), (64, 52), (65, 53)], 1, 1)

# White highlights
c4.curve([(61, 51), (62, 52)], 1, 1)
c4.curve([(66, 53), (67, 54)], 1, 1)
c4.curve([(64, 52)], 1, 1)
c4.curve([(48, 46), (50, 48)], 1, 1)
c4.curve([(80, 48), (78, 50)], 1, 1)

# Black shadows
c4.curve([(17, 28), (15, 34)], 0, 1)
c4.curve([(15, 55), (13, 62)], 0, 1)
c4.curve([(111, 30), (113, 36)], 0, 1)
c4.curve([(113, 57), (115, 64)], 0, 1)

c4.save_png('/home/user/tetrahedral-nn/portraits/bridge_4.png')

print("✓ Stages complete")
