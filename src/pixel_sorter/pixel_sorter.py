from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class SortDirection(Enum):
    LEFT_TO_RIGHT = "left-to-right"
    RIGHT_TO_LEFT = "right-to-left"
    TOP_TO_BOTTOM = "top-to-bottom"
    BOTTOM_TO_TOP = "bottom-to-top"


class SortMode(Enum):
    R = "R"
    G = "G"
    B = "B"
    GRAYSCALE = "grayscale"
    LUMINANCE = "luminance"
    HUE = "hue"
    SATURATION = "saturation"
    VALUE = "value"


class SortFunctions:
    @staticmethod
    def sort_luminance(segment: np.ndarray) -> np.ndarray:
        rgb = segment.astype(np.float32) / 255.0
        return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    @staticmethod
    def sort_hue(segment: np.ndarray) -> np.ndarray:
        rgb = segment.astype(np.float32) / 255.0
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        maxc = np.maximum.reduce([r, g, b])
        minc = np.minimum.reduce([r, g, b])
        delta = (maxc - minc) + 0.0001

        h = np.zeros_like(maxc)
        mask = delta != 0

        rc = ((g - b) / delta) % 6
        gc = ((b - r) / delta) + 2
        bc = ((r - g) / delta) + 4

        rmask = (maxc == r) & mask
        gmask = (maxc == g) & mask
        bmask = (maxc == b) & mask

        h[rmask] = rc[rmask]
        h[gmask] = gc[gmask]
        h[bmask] = bc[bmask]

        h = (h / 6.0) % 1.0

        return h

    @staticmethod
    def sort_saturation(segment: np.ndarray) -> np.ndarray:
        rgb = segment.astype(np.float32) / 255.0
        maxc = np.max(rgb, axis=1)
        minc = np.min(rgb, axis=1)
        delta = maxc - minc
        s = np.zeros_like(maxc)
        mask = maxc != 0
        s[mask] = delta[mask] / maxc[mask]
        return s

    @staticmethod
    def sort_value(segment: np.ndarray) -> np.ndarray:
        rgb = segment.astype(np.float32) / 255.0
        return np.max(rgb, axis=1)

    functions = {
        SortMode.R: lambda seg: seg[:, 0] / 255.0,
        SortMode.G: lambda seg: seg[:, 1] / 255.0,
        SortMode.B: lambda seg: seg[:, 2] / 255.0,
        SortMode.GRAYSCALE: lambda seg: seg[:, :3].mean(axis=1) / 255.0,
        SortMode.LUMINANCE: sort_luminance,
        SortMode.HUE: sort_hue,
        SortMode.SATURATION: sort_saturation,
        SortMode.VALUE: sort_value,
    }


class PixelSorter:
    def __init__(self, path: str) -> None:
        self.pil_image = Image.open(path)
        # self.pil_image = ImageOps.exif_transpose(self.pil_image)

        self.sort_functions = SortFunctions.functions

        self.mask_image: Image.Image | None = None
        self.sorted_image: Image.Image | None = None

    def create_mask(self, mask_function: SortMode, mask_threshold: float, invert: bool) -> None:
        img = np.array(self.pil_image).astype(np.float32)
        h, w, _ = img.shape
        flat_pixels = img.reshape(-1, 3)

        values = self.sort_functions[mask_function](flat_pixels) * 255.0
        mask = (values < mask_threshold).reshape(h, w)
        if invert:
            mask = ~mask

        self.mask_image = Image.fromarray(mask)

    def create_segments(self, mask):
        a = np.pad(mask, ((0, 0), (0, 1)), constant_values=0)
        b = np.pad(mask, ((0, 0), (1, 0)), constant_values=0)

        starts = a & ~b
        ends = ~a & b
        sx, sy = np.nonzero(starts)
        ex, ey = np.nonzero(ends)

        return (sx, sy, ex, ey)

    def sort(self, mode: SortMode, direction: SortDirection) -> None:
        if not self.pil_image or not self.mask_image:
            return

        img_array = np.asarray(self.pil_image).copy()

        sort_function = self.sort_functions[mode]

        is_vertical = direction in [
            SortDirection.TOP_TO_BOTTOM,
            SortDirection.BOTTOM_TO_TOP,
        ]
        is_reverse = direction in [
            SortDirection.RIGHT_TO_LEFT,
            SortDirection.BOTTOM_TO_TOP,
        ]

        if is_vertical:
            img_array = img_array.transpose((1, 0, 2))  # (width, height, rgb)
            mask_array = np.asarray(self.mask_image).T
        else:
            mask_array = np.asarray(self.mask_image)

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
