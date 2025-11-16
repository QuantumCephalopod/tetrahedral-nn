#!/usr/bin/env python3
"""
Painting tools: Canvas + Brushes
Define forms programmatically, experiment dynamically
"""
import math
import struct
import zlib

class Canvas:
    """Rectangular canvas with 4-color palette"""

    # Black, White, Orange, Blue
    PALETTE = [
        (0, 0, 0),        # 0: pure black
        (255, 255, 255),  # 1: pure white
        (255, 140, 0),    # 2: orange (warm)
        (0, 100, 255)     # 3: blue (cool)
    ]

    PALETTE_CHARS = ['█', '░', '▓', '▒']  # For text representation

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]  # Start with black

    def set_pixel(self, x, y, color_idx):
        """Set pixel color (0=black, 1=white, 2=orange, 3=blue)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = max(0, min(3, color_idx))

    def get_pixel(self, x, y):
        """Get pixel color index"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return 0

    def circle(self, cx, cy, radius, color_idx):
        """Paint a circle"""
        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius:
                    self.set_pixel(x, y, color_idx)

    def line(self, x0, y0, x1, y1, color_idx, thickness=1):
        """Paint a line"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Paint with thickness
            for dt in range(-thickness//2, thickness//2 + 1):
                for dt2 in range(-thickness//2, thickness//2 + 1):
                    self.set_pixel(x0 + dt, y0 + dt2, color_idx)

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def ellipse(self, cx, cy, rx, ry, color_idx):
        """Paint an ellipse"""
        for y in range(self.height):
            for x in range(self.width):
                # Ellipse equation: (x-cx)²/rx² + (y-cy)²/ry² < 1
                if ((x - cx)**2 / (rx**2 + 0.01) + (y - cy)**2 / (ry**2 + 0.01)) < 1:
                    self.set_pixel(x, y, color_idx)

    def to_text(self):
        """Convert to text representation"""
        return '\n'.join(''.join(self.PALETTE_CHARS[cell] for cell in row) for row in self.grid)

    def save_text(self, filename):
        """Save as text file"""
        with open(filename, 'w') as f:
            f.write(self.to_text())

    def curve(self, points, color_idx, thickness=2):
        """Paint a smooth curve through points [(x1,y1), (x2,y2), ...]"""
        if len(points) < 2:
            return

        # Draw smooth curve through points
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            # Interpolate between points
            steps = int(math.sqrt((x1-x0)**2 + (y1-y0)**2))
            for step in range(steps):
                t = step / max(steps, 1)
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))

                # Paint with thickness
                for dx in range(-thickness, thickness + 1):
                    for dy in range(-thickness, thickness + 1):
                        if dx*dx + dy*dy <= thickness*thickness:
                            self.set_pixel(x + dx, y + dy, color_idx)

    def save_png(self, filename, scale=3):
        """Save as RGB PNG"""
        def chunk(typ, data):
            return struct.pack('>I', len(data)) + typ + data + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff)

        png = b'\x89PNG\r\n\x1a\n'
        # Color type 2 = RGB (8 bits per channel)
        ihdr = struct.pack('>IIBBBBB', self.width * scale, self.height * scale, 8, 2, 0, 0, 0)
        png += chunk(b'IHDR', ihdr)

        raw = b''
        for row in self.grid:
            for _ in range(scale):
                raw += b'\x00'  # Filter type 0 (no filter)
                for cell in row:
                    r, g, b = self.PALETTE[cell]
                    # Repeat pixel 'scale' times horizontally
                    for _ in range(scale):
                        raw += bytes([r, g, b])

        png += chunk(b'IDAT', zlib.compress(raw, 1))
        png += chunk(b'IEND', b'')

        with open(filename, 'wb') as f:
            f.write(png)


def composite_stages(canvases, output_file, scale=3):
    """Combine 4 canvases into 2x2 grid composite"""
    if len(canvases) != 4:
        raise ValueError("Need exactly 4 canvases")

    w, h = canvases[0].width, canvases[0].height

    # Create composite canvas
    comp = Canvas(w * 2, h * 2)

    # Copy each stage into quadrant
    for idx, canvas in enumerate(canvases):
        x_offset = (idx % 2) * w
        y_offset = (idx // 2) * h

        for y in range(h):
            for x in range(w):
                comp.set_pixel(x + x_offset, y + y_offset, canvas.get_pixel(x, y))

    comp.save_png(output_file, scale)
    return comp


if __name__ == '__main__':
    # Test the color canvas
    c = Canvas(60, 30)
    c.circle(30, 15, 10, 2)  # Orange circle
    c.save_png('test_color.png')
    print("✓ Saved test_color.png")
