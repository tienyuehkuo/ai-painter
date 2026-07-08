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
