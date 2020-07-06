from PIL import Image
from typing import Union


class Box:
    """
    (x_0, y_0) ----- O
        .            .
        .            .
        .            .
        .            .
        O  ---- (x_1, y_1)
    """
    def __init__(self, x0: Union[int, float], y0: Union[int, float], x1: Union[int, float], y1: Union[int, float]):

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class Quad:
    """
    (x_0, y_0) --------- (x_3, y_3)
         .                    .
         .                    .
         .                    .
         .                    .
    (x_1, y_1) ---------- (x_2, y_2)
    """
    def __init__(self,
                 x0: Union[int, float], y0: Union[int, float],
                 x1: Union[int, float], y1: Union[int, float],
                 x2: Union[int, float], y2: Union[int, float],
                 x3: Union[int, float], y3: Union[int, float]):

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3


class Grid:

    def __init__(self, height: int, width: int, pixel_height: int, pixel_width: int)

        self.height = height
        self.width = width
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width

    @classmethod
    def from_image(cls, image: Image, height: int, width: int):

        pixel_width, pixel_height = image.size
        return cls(height, width, pixel_width, pixel_height)

