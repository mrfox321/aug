from PIL import Image
from typing import Union, List, Tuple, Optional
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

    @classmethod
    def from_corner(cls, grid: np.ndarray, i: int, j: int):
        return cls(grid[i, j, 0], grid[i, j, 1], grid[i + 1, j + 1, 0], grid[i + 1, j + 1, 1])


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

    @classmethod
    def from_corner(cls, grid: np.ndarray, i: int, j: int):
        return cls(grid[i, j, 0], grid[i, j, 1],
                   grid[i + 1, j, 0], grid[i + 1, j, 1],
                   grid[i + 1, j + 1, 0], grid[i + 1, j + 1, 1],
                   grid[i, j + 1, 0], grid[i, j + 1, 1])


def build_grid(height: int, width: int, pixel_height: int, pixel_width: int) -> np.ndarray:
    """
    arr[i, j] = [pixel_height / height * i, pixel_width / width * j]
    """
    heights = pixel_height / height * np.arange(height + 1)
    widths = pixel_width / width * np.arange(width + 1)
    return np.stack(np.meshgrid(heights, widths, indexing='ij'), axis=-1)


def jitter(grid: np.ndarray, scale: float, is_pinned=True) -> np.ndarray:
    """
    Add gaussian noise to regular grid.  Boolean flag pins the boundary of the grid
    """
    rng = np.random.normal(0, scale, grid.shape)
    if is_pinned:
        rng[0] = rng[-1] = rng[:, 0] = rng[:, -1] = 0.
    return grid + rng


def to_boxes(grid: np.ndarray) -> List[Box]:

    height, width, _ = grid.shape
    boxes = [Box.from_corner(grid, i, j) for i in range(height - 1) for j in range(width - 1)]
    return boxes


def to_quads(grid: np.ndarray) -> List[Quad]:
    height, width, _ = grid.shape
    quads = [Quad.from_corner(grid, i, j) for i in range(height - 1) for j in range(width - 1)]
    return quads


class Grid:

    def __init__(self, grid: np.ndarray):

        self.grid = grid

    @property
    def height(self):
        return self.grid.shape[0] - 1

    @property
    def width(self):
        return self.grid.shape[1] - 1

    @property
    def pixel_height(self):
        return self.grid[-1, 0, 0]

    @property
    def pixel_width(self):
        return self.grid[0, -1, 1]

    @classmethod
    def from_image(cls, image: Image, height: int, width: int):

        pixel_width, pixel_height = image.size
        grid = build_grid(height, width, pixel_height, pixel_width)
        return cls(grid)


def augment(image: Image, height: int, width: int, scale: float) -> Image:

    grid = Grid.from_image(image, height, width)


############  TESTS   ################

def test_grid():

    path = 'test.png'
    image = Image.open(path)
    height = width = 10
    pixel_width, pixel_height = image.size

    grid = Grid(build_grid(height, width, pixel_height, pixel_width))

    assert grid.height == height
    assert grid.width == width

    assert grid.pixel_width == pixel_width
    assert grid.pixel_height == pixel_height