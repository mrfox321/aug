import cv2
from PIL import Image
from typing import Union, List, Tuple, Optional, Sequence, Iterator
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
    def from_image(cls, image: Image.Image, height: int, width: int):
        pixel_width, pixel_height = image.size
        grid = build_grid(height, width, pixel_height, pixel_width)
        return cls(grid)


def build_mesh(boxes: List[Box], quads: List[Quad]) -> List[List[Sequence[float]]]:
    mesh = [[box.as_tuple(), quad.as_list()] for box, quad in zip(boxes, quads)]
    return mesh


def image2grid(image: Image.Image, height: int, width: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    From an image generate the underlying regular grid and some random initialization
    """
    grid = Grid.from_image(image, height, width)
    scale_width, scale_height = scale * grid.block_width, scale * grid.block_height
    random_grid = jitter(grid.grid, scale_width, scale_height)
    return grid.grid, random_grid


def jitter_image(image: Image.Image, height: int, width: int, scale: float, resample=Image.NEAREST) -> Image.Image:
    """

    :param image: PIL Image Object
    :param height: number of coarse grained pixels in y-direction
    :param width: number of coarse grained pixels in x-direction
    :param scale: std-dev of fluctuations (scale := pixel scale / block size)
    :param resample: resample method for PIL.Image
    :return:
    """
    grid, random_grid = image2grid(image, height, width, scale)

    boxes = to_boxes(grid)
    quads = to_quads(random_grid)
    mesh = build_mesh(boxes, quads)

    random_image = image.transform(image.size, Image.MESH, mesh, resample)
    return random_image


def jitter_image_array(arr: np.ndarray, height: int, width: int, scale: float, resample=Image.NEAREST) -> np.ndarray:
    """
    numpy bitmap -> numpy bitmap transformation of `jitter_image`
    """
    image = Image.fromarray(arr)
    rng_image = jitter_image(image, height, width, scale, resample)
    rng_arr = np.array(rng_image)
    return rng_arr


class MeshIter:

    def __init__(self, grid_base: np.ndarray, grid_init: np.ndarray,
                 delta_t: float, v_init: Optional[np.ndarray] = None):

        self.grid_base = grid_base                         # array of equilibrium positions
        self.grid_init = grid_init                         # array of initial positions
        self.mesh: deque[np.ndarray] = deque([])           # array of displacements from equilibrium
        self.v_init = v_init or np.zeros_like(grid_base)   # array of initial velocities
        self.delta_t = delta_t                             # time step for verlet integration

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

    def __next__(self) -> np.ndarray:

        if len(self.mesh) == 0:
            self.mesh.append(self.grid_init - self.grid_base)
        elif len(self.mesh) == 1:
            next_grid = self.init_step()
            self.mesh.append(next_grid)
        else:
            next_grid = self.step()
            self.mesh.append(next_grid)
            self.mesh.popleft()
        return self.mesh[-1] + self.grid_base

    @classmethod
    def from_image(cls, image: Image, height: int, width: int, scale: float,
                   delta_t: float, v_init: Optional[np.ndarray] = None):
        grid_base, random_grid = image2grid(image, height, width, scale)
        return cls(grid_base, random_grid, delta_t, v_init)


def frame_iter(meshiter: MeshIter, image: Image, resample=Image.NEAREST) -> Iterator[Image.Image]:

    boxes = to_boxes(meshiter.grid_base)
    for grid in meshiter:
        quads = to_quads(grid)
        mesh = build_mesh(boxes, quads)
        yield image.transform(image.size, Image.MESH, mesh, resample)


class ImageAugment:

    def __init__(self, height: int, width: int, scale: float, resample=Image.NEAREST):

        self.height = height
        self.width = width
        self.scale = scale
        self.resample = resample

    def augment(self, arr: np.ndarray):
        return jitter_image_array(arr, self.height, self.width, self.scale, self.resample)


def image_to_cv2_array(image: Image) -> np.ndarray:

    open_cv_image_RGB_array = np.array(image)
    # RGB -> BGR
    open_cv_image_BGR_array = open_cv_image_RGB_array[:, :, ::-1]
    return open_cv_image_BGR_array


class Frames:

    def __init__(self, frames: Iterator[Image.Image], image: Image.Image, fps: float, path: str):

        self.cv2_frames = map(image_to_cv2_array, frames)
        self.image = image
        self.fps = fps
        self.name = path + '.mp4'
        self.video = cv2.VideoWriter(self.name, cv2.VideoWriter_fourcc(*'MP4V'), fps, image.size)

    @property
    def width(self):
        return self.image.size[0]

    @property
    def height(self):
        return self.image.size[1]

    def write(self, n_frames: int):
        for _ in range(n_frames):
            self.video.write(next(self.cv2_frames))
        cv2.destroyAllWindows()
        self.video.release()

    @classmethod
    def from_random_image(cls, image_path: str, video_path: str, fps: float, height: int = 10, width: int = 20,
                          scale: float = 0.1, delta_t: float = 0.5, v_init: Optional[np.ndarray] = None):
        image = Image.open(image_path)
        meshiter = MeshIter.from_image(image, height, width, scale, delta_t, v_init)
        frames = frame_iter(meshiter, image)
        return cls(frames, image, fps, video_path)


def write_random_video(
        src: str,
        dest: str,
        n_frames: int,
        fps: float,
        height: int,
        width: int,
        scale: float,
        delta_t: float
):
    Frames.from_random_image(src, dest, fps, height, width, scale, delta_t).write(n_frames)


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
