import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from enum import Enum

class SortDirection(Enum):
    LEFT_TO_RIGHT = "left-to-right"
    RIGHT_TO_LEFT = "right-to-left"
    TOP_TO_BOTTOM = "top-to-bottom"
    BOTTOM_TO_TOP = "bottom-to-top"


class SortMode(Enum):
    R = "R"
    G = "G"
    B = "B"
    LUMINANCE = "luminance"
    HUE = "hue"


class SortFunctions:
    def sort_r(segment: np.ndarray) -> np.ndarray:
        return segment[:, 0]

    def sort_g(segment: np.ndarray) -> np.ndarray:
        return segment[:, 1]

    def sort_b(segment: np.ndarray) -> np.ndarray:
        return segment[:, 2]

    def sort_luminance(segment: np.ndarray) -> np.ndarray:
        return 0.299 * segment[:, 0] + 0.587 * segment[:, 1] + 0.114 * segment[:, 2]

    def sort_hue(segment: np.ndarray) -> np.ndarray:
        norm = segment / 255.0
        return np.array([colorsys.rgb_to_hsv(*pixel)[0] for pixel in norm])
    
    functions = {
        SortMode.R: sort_r,
        SortMode.G: sort_g,
        SortMode.B: sort_b,
        SortMode.LUMINANCE: sort_luminance,
        SortMode.HUE: sort_hue,
    }

class PixelSorter:
    def __init__(self, path: str, mask_threshold: float) -> None:
        self.pil_image = Image.open(path)
        self.pil_image = ImageOps.exif_transpose(self.pil_image)

        self.mask_threshold = mask_threshold
        self.mask = self.create_mask()

        self.sorted_image: Image.Image | None = None

        self.sort_functions = SortFunctions.functions

    def set_mask_threshold(self, mask_threshold: float):
        self.mask_threshold = mask_threshold
        self.mask = self.create_mask()

    def create_mask(self) -> Image.Image:
        mask = self.pil_image.convert("L")
        mask = mask.point(lambda p: 255 if p > self.mask_threshold else 0)
        mask = mask.convert("1")
        return mask

    def create_segments(self, mask):
        a = np.pad(mask, ((0, 0), (0, 1)), constant_values=0)
        b = np.pad(mask, ((0, 0), (1, 0)), constant_values=0)

        starts = a & ~b
        ends = ~a & b
        sx, sy = np.nonzero(starts)
        ex, ey = np.nonzero(ends)

        return (sx, sy, ex, ey)

    def sort(self, mode: SortMode, direction: SortDirection) -> Image.Image:
        img_array = np.asarray(self.pil_image).copy()

        sort_function = self.sort_functions[mode]

        is_vertical = direction in [SortDirection.TOP_TO_BOTTOM, SortDirection.BOTTOM_TO_TOP]
        is_reverse = direction in [SortDirection.RIGHT_TO_LEFT, SortDirection.BOTTOM_TO_TOP]

        if is_vertical:
            img_array = img_array.transpose((1, 0, 2))  # (width, height, rgb)
            mask_array = np.asarray(self.mask).T
        else:
            mask_array = np.asarray(self.mask)

        sx, sy, _, ey = self.create_segments(mask_array)
        for i in range(sx.shape[0]):
            x = sx[i]
            start = sy[i]
            end = ey[i]
            segment = img_array[x, start:end]

            if start >= end:
                continue

            values = sort_function(segment)
            argsort = np.argsort(values)
            if is_reverse:
                argsort = argsort[::-1]

            img_array[x, start:end] = segment[argsort]

        if is_vertical:
            img_array = img_array.transpose((1, 0, 2))

        self.sorted_image = Image.fromarray(img_array)

    def save(self, path: str):
        if self.sorted_image:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.sorted_image.save(path)
