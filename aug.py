from PIL import Image
from typing import Union, List, Tuple, Optional, Sequence
import numpy as np
from scipy.signal import convolve2d
from collections import deque


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


def zero_edges(arr: np.ndarray) -> np.ndarray:
    arr[0] = arr[-1] = arr[:, 0] = arr[:, -1] = 0.
    return arr


class Physics:

    @staticmethod
    def accelerate(grid: np.ndarray, is_pinned: bool = True) -> np.ndarray:

        coupling_x, coupling_y = np.array([[1, -2, 1]]), np.array([[1, -2, 1]]).T

        acc = np.zeros_like(grid)
        acc[..., 0] = convolve2d(grid[..., 0], coupling_x, mode='same')
        acc[..., 1] = convolve2d(grid[..., 1], coupling_y, mode='same')
        if is_pinned:
            acc = zero_edges(acc)
        return acc


def build_grid(height: int, width: int, pixel_height: int, pixel_width: int) -> np.ndarray:
    """
    arr[i, j] = [pixel_height / height * i, pixel_width / width * j]

    NOTE:
    Integer array *required* for PIL API (`box` tuple/list)
    """
    heights = pixel_height / height * np.arange(height + 1)
    widths = pixel_width / width * np.arange(width + 1)
    return np.stack(np.meshgrid(widths, heights, indexing='xy'), axis=-1).astype(np.int)


def jitter(grid: np.ndarray, scale_x: float, scale_y: Optional[float] = None, is_pinned=True) -> np.ndarray:
    """
    Add gaussian noise to regular grid.  Boolean flag pins the boundary of the grid
    """
    if scale_y:
        rng = [np.random.normal(0, scale_x, grid.shape[:-1]), np.random.normal(0, scale_y, grid.shape[:-1])]
        rng = np.stack(rng, axis=-1)
    else:
        rng = np.random.normal(0, scale_x, grid.shape)
    if is_pinned:
        rng = zero_edges(rng)
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
        return self.grid[-1, 0, 1]

    @property
    def pixel_width(self):
        return self.grid[0, -1, 0]

    @property
    def block_height(self):
        return self.grid[1, 0, 1] - self.grid[0, 0, 1]

    @property
    def block_width(self):
        return self.grid[0, 1, 0] - self.grid[0, 0, 0]

    @classmethod
    def from_image(cls, image: Image, height: int, width: int):
        pixel_width, pixel_height = image.size
        grid = build_grid(height, width, pixel_height, pixel_width)
        return cls(grid)


def build_mesh(boxes: List[Box], quads: List[Quad]) -> List[List[Sequence[float]]]:
    mesh = [[box.as_tuple(), quad.as_list()] for box, quad in zip(boxes, quads)]
    return mesh


def jitter_image(image: Image, height: int, width: int, scale: float, resample=Image.NEAREST) -> Image:
    """

    :param image: PIL Image Object
    :param height: number of coarse grained pixels in y-direction
    :param width: number of coarse grained pixels in x-direction
    :param scale: std-dev of fluctuations (scale := pixel scale / block size)
    :param resample: resample method for PIL.Image
    :return:
    """
    grid = Grid.from_image(image, height, width)
    scale_width, scale_height = scale * grid.block_width, scale * grid.block_height

    random_grid = jitter(grid.grid, scale_width, scale_height)

    boxes = to_boxes(grid.grid)
    quads = to_quads(random_grid)
    mesh = build_mesh(boxes, quads)

    random_image = image.transform(image.size, Image.MESH, mesh, resample)
    return random_image


class MeshIter:

    def __init__(self, grid: Grid, v_init: np.ndarray, delta_t: float):

        self.grid = grid
        self.mesh: deque[np.ndarray] = deque([])
        self.v_init = v_init
        self.delta_t = delta_t

    def __iter__(self):
        return self

    @property
    def acceleration(self) -> np.ndarray:
        return Physics.accelerate(self.mesh[-1])

    def init_step(self) -> np.ndarray:
        next_grid = self.mesh[0] + self.v_init * self.delta_t + 0.5 * self.acceleration * self.delta_t ** 2
        return next_grid

    def step(self) -> np.ndarray:
        next_grid = 2 * self.mesh[1] - self.mesh[0] + self.acceleration * self.delta_t ** 2
        return next_grid

    def __next__(self):

        if len(self.mesh) == 0:
            self.mesh.append(self.grid.grid)
        elif len(self.mesh) == 1:
            next_grid = self.init_step()
            self.mesh.append(next_grid)
        else:
            next_grid = self.step()
            self.mesh.append(next_grid)
            self.mesh.popleft()
        return self.mesh[-1]


############  TESTS   ################


def build_grid_impl(height: int, width: int, pixel_height: int, pixel_width: int) -> np.ndarray:
    heights = pixel_height / height * np.arange(height + 1)
    widths = pixel_width / width * np.arange(width + 1)
    grid = np.zeros((height+1, width+1, 2))
    for i, h in enumerate(heights):
        for j, w in enumerate(widths):
            grid[i, j] = [w, h]
    return grid.astype(np.int)


def test_grid():
    path = 'test.png'
    image = Image.open(path)
    height = width = 10
    pixel_width, pixel_height = image.size

    grid_arr = build_grid(height, width, pixel_height, pixel_width)
    grid_impl_arr = build_grid_impl(height, width, pixel_height, pixel_width)

    assert np.allclose(grid_arr, grid_impl_arr)

    grid = Grid(grid_arr)

    assert grid.height == height
    assert grid.width == width

    assert grid.pixel_width == pixel_width
    assert grid.pixel_height == pixel_height

    assert grid.block_height == pixel_height / height
    assert grid.block_width == pixel_width / width
