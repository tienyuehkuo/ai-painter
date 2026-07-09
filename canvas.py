import math

from PIL import Image


class Canvas:
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._pixels = [
            [(255, 255, 255) for x in range(width)]
            for y in range(height)
        ]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def pixels(self):
        return self._pixels

    def to_image(self):
        image = Image.new("RGB", (self._width, self._height))
        image.putdata([pixel for row in self._pixels for pixel in row])
        return image

    def save(self, filename):
        self.to_image().save(filename)

    def draw_stroke(self, x1, y1, x2, y2, radius, color):
        for y in range(self._height):
            for x in range(self._width):
                distance = self._distance_to_line_segment(x, y, x1, y1, x2, y2)

                if distance <= radius:
                    self._pixels[y][x] = color

    def draw_normalized_stroke(self, stroke, min_radius=1, max_radius=None):
        if max_radius is None:
            max_radius = max(self._width, self._height)

        x1 = stroke[0] * (self._width - 1)
        y1 = stroke[1] * (self._height - 1)
        x2 = stroke[2] * (self._width - 1)
        y2 = stroke[3] * (self._height - 1)
        radius = min_radius + stroke[4] * (max_radius - min_radius)
        color = (
            int(round(stroke[5] * 255)),
            int(round(stroke[6] * 255)),
            int(round(stroke[7] * 255)),
        )

        self.draw_stroke(x1, y1, x2, y2, radius, color)

    def _distance_to_line_segment(self, px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
