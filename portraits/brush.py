#!/usr/bin/env python3
"""
Painting tools: Canvas + Brushes
Define forms programmatically, experiment dynamically
"""
import math
import struct
import zlib

class Canvas:
    """Rectangular canvas with 4-tone palette"""

    PALETTE = ['█', '▓', '▒', '░', ' ']  # 0-4 intensity

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[4 for _ in range(width)] for _ in range(height)]  # Start with space (4)

    def set_pixel(self, x, y, intensity):
        """Set pixel intensity (0=█ brightest, 4=space darkest)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = max(0, min(4, intensity))

    def get_pixel(self, x, y):
        """Get pixel intensity"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return 4

    def circle(self, cx, cy, radius, intensity, gradient=True):
        """Paint a circle with optional gradient"""
        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)

                if gradient:
                    # Gradient based on distance
                    if dist < radius * 0.3:
                        self.set_pixel(x, y, min(self.get_pixel(x, y), intensity))
                    elif dist < radius * 0.5:
                        self.set_pixel(x, y, min(self.get_pixel(x, y), intensity + 1))
                    elif dist < radius * 0.7:
                        self.set_pixel(x, y, min(self.get_pixel(x, y), intensity + 2))
                    elif dist < radius:
                        self.set_pixel(x, y, min(self.get_pixel(x, y), intensity + 3))
                else:
                    if dist < radius:
                        self.set_pixel(x, y, min(self.get_pixel(x, y), intensity))

    def line(self, x0, y0, x1, y1, intensity, thickness=1):
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
                    self.set_pixel(x0 + dt, y0 + dt2, min(self.get_pixel(x0 + dt, y0 + dt2), intensity))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def torus(self, cx, cy, outer_r, inner_r, intensity, gradient=True):
        """Paint a torus (donut shape)"""
        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)

                if inner_r < dist < outer_r:
                    if gradient:
                        # Gradient based on distance from center of torus band
                        mid = (inner_r + outer_r) / 2
                        rel_dist = abs(dist - mid) / ((outer_r - inner_r) / 2)
                        i = int(intensity + rel_dist * 3)
                        self.set_pixel(x, y, min(self.get_pixel(x, y), i))
                    else:
                        self.set_pixel(x, y, min(self.get_pixel(x, y), intensity))

    def to_text(self):
        """Convert to text representation"""
        return '\n'.join(''.join(self.PALETTE[cell] for cell in row) for row in self.grid)

    def save_text(self, filename):
        """Save as text file"""
        with open(filename, 'w') as f:
            f.write(self.to_text())

    def curve(self, points, intensity, thickness=2):
        """Paint a smooth curve through points [(x1,y1), (x2,y2), ...]"""
        if len(points) < 2:
            return

        # Draw smooth curve through points using Catmull-Rom spline
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
                            self.set_pixel(x + dx, y + dy, min(self.get_pixel(x + dx, y + dy), intensity))

    def save_png(self, filename, scale=3):
        """Save as PNG"""
        char_map = {'█': 255, '▓': 200, '▒': 140, '░': 80, ' ': 0}

        def chunk(typ, data):
            return struct.pack('>I', len(data)) + typ + data + struct.pack('>I', zlib.crc32(typ + data) & 0xffffffff)

        png = b'\x89PNG\r\n\x1a\n'
        ihdr = struct.pack('>IIBBBBB', self.width * scale, self.height * scale, 8, 0, 0, 0, 0)
        png += chunk(b'IHDR', ihdr)

        raw = b''
        for row in self.grid:
            for _ in range(scale):
                raw += b'\x00'
                for cell in row:
                    val = char_map[self.PALETTE[cell]]
                    raw += bytes([val] * scale)

        png += chunk(b'IDAT', zlib.compress(raw, 9))
        png += chunk(b'IEND', b'')

        with open(filename, 'wb') as f:
            f.write(png)


if __name__ == '__main__':
    # Test the canvas
    c = Canvas(60, 30)

    # Paint two toruses - closed and broken
    c.torus(15, 15, 10, 6, 0, gradient=True)
    c.torus(45, 15, 10, 6, 0, gradient=True)

    # Break the right one (paint over with background)
    for y in range(10, 20):
        for x in range(40, 43):
            c.set_pixel(x, y, 4)

    print(c.to_text())
    c.save_text('test_canvas.txt')
    c.save_png('test_canvas.png')
    print("\n✓ Saved test_canvas.txt and test_canvas.png")
