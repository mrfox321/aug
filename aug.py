from PIL import Image
from typing import Union, List, Tuple
import numpy as np


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

    def as_tuple(self) -> Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]]:
        return self.x0, self.y0, self.x1, self.y1

    def as_list(self) -> Union[List[float], List[int]]:
        return list(self.as_tuple())


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

    def as_tuple(self) -> Tuple:
        return self.x0, self.y0, self.x1, self.y1, self.x2, self.y2, self.x3, self.y3

    def as_list(self) -> List[Union[int, float]]:
        return list(self.as_tuple())


class Grid:

    def __init__(self, height: int, width: int, pixel_height: int, pixel_width: int):

        self.height = height
        self.width = width
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width
        self.grid = self.build_grid(height, width, pixel_height, pixel_width)

    @staticmethod
    def build_grid(height: int, width: int, pixel_height: int, pixel_width: int) -> np.ndarray:
        """
        arr[i, j] = [pixel_height / height * i, pixel_width / width * j]
        """
        heights = pixel_height / height * np.arange(height + 1)
        widths = pixel_width / width * np.arange(width + 1)
        return np.stack(np.meshgrid(heights, widths, indexing='ij'), axis=-1)

    @classmethod
    def from_image(cls, image: Image, height: int, width: int):

        pixel_width, pixel_height = image.size
        return cls(height, width, pixel_width, pixel_height)

