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

    def save(self, filename):
        image = Image.new("RGB", (self._width, self._height))
        image.putdata([pixel for row in self._pixels for pixel in row])
        image.save(filename)
