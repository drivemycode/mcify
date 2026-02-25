# mcify_numba.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
from numba import njit
from pathlib import Path

import minecraft_blocks

RGB = Tuple[int, int, int]

# ----------------------------
# Palette (Python-side)
# ----------------------------

@dataclass(frozen=True)
class Palette:
    names: List[str]         # length K
    rgb: np.ndarray          # (K,3) float32

    @staticmethod
    def from_blocks(blocks: Dict[str, RGB]) -> "Palette":
        names = list(blocks.keys())
        rgb = np.array([blocks[n] for n in names], dtype=np.float32)
        return Palette(names=names, rgb=rgb)


def load_image_rgb(path: str, out_size: Tuple[int, int], resize_mode: int = Image.BILINEAR) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(out_size, resize_mode)
    return np.array(img, dtype=np.float32)


def preview_from_indices(idx: np.ndarray, palette: Palette) -> Image.Image:
    rgb_u8 = palette.rgb.astype(np.uint8)
    return Image.fromarray(rgb_u8[idx], mode="RGB")


# ----------------------------
# Numba core
# ----------------------------

@njit(cache=True, fastmath=True)
def _nearest_index_rgb(color_rgb, palette_rgb):
    best_i = 0
    best_d = 1e30
    cr0, cr1, cr2 = color_rgb[0], color_rgb[1], color_rgb[2]
    for i in range(palette_rgb.shape[0]):
        p0, p1, p2 = palette_rgb[i, 0], palette_rgb[i, 1], palette_rgb[i, 2]
        d0 = p0 - cr0
        d1 = p1 - cr1
        d2 = p2 - cr2
        d = d0*d0 + d1*d1 + d2*d2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


@njit(cache=True, fastmath=True)
def dither_fs_serpentine_numba(img_rgb_float, palette_rgb_float, clamp_enabled=True):
    arr = img_rgb_float.copy()
    H, W, _ = arr.shape
    idx = np.zeros((H, W), dtype=np.int32)

    for y in range(H):
        if y % 2 == 0:
            x_start, x_end, step = 0, W, 1
            dirx = 1
        else:
            x_start, x_end, step = W - 1, -1, -1
            dirx = -1

        x = x_start
        while x != x_end:
            if clamp_enabled:
                for c in range(3):
                    v = arr[y, x, c]
                    if v < 0.0:
                        arr[y, x, c] = 0.0
                    elif v > 255.0:
                        arr[y, x, c] = 255.0

            old0, old1, old2 = arr[y, x, 0], arr[y, x, 1], arr[y, x, 2]
            k = _nearest_index_rgb(arr[y, x], palette_rgb_float)
            idx[y, x] = k

            new0, new1, new2 = palette_rgb_float[k, 0], palette_rgb_float[k, 1], palette_rgb_float[k, 2]
            err0, err1, err2 = old0 - new0, old1 - new1, old2 - new2

            x1 = x + dirx
            if 0 <= x1 < W:
                arr[y, x1, 0] += err0 * (7.0 / 16.0)
                arr[y, x1, 1] += err1 * (7.0 / 16.0)
                arr[y, x1, 2] += err2 * (7.0 / 16.0)

            if y + 1 < H:
                arr[y + 1, x, 0] += err0 * (5.0 / 16.0)
                arr[y + 1, x, 1] += err1 * (5.0 / 16.0)
                arr[y + 1, x, 2] += err2 * (5.0 / 16.0)

                xl = x - dirx
                xr = x + dirx
                if 0 <= xl < W:
                    arr[y + 1, xl, 0] += err0 * (3.0 / 16.0)
                    arr[y + 1, xl, 1] += err1 * (3.0 / 16.0)
                    arr[y + 1, xl, 2] += err2 * (3.0 / 16.0)
                if 0 <= xr < W:
                    arr[y + 1, xr, 0] += err0 * (1.0 / 16.0)
                    arr[y + 1, xr, 1] += err1 * (1.0 / 16.0)
                    arr[y + 1, xr, 2] += err2 * (1.0 / 16.0)

            x += step

    return idx


# ----------------------------
# End-to-end wrapper
# ----------------------------

def mosaic_dither_numba(
    image_path: str,
    blocks: Dict[str, RGB],
    out_size: Tuple[int, int] = (128, 128),
    resize_mode: int = Image.BILINEAR,
    clamp: bool = True,
):
    """
    End-to-end: load image -> build palette -> dither (Numba) -> return (grid, idx, preview_img)
    """
    palette = Palette.from_blocks(blocks)
    img = load_image_rgb(image_path, out_size, resize_mode=resize_mode)

    # Call JIT core
    idx = dither_fs_serpentine_numba(img.astype(np.float32), palette.rgb.astype(np.float32), clamp_enabled=clamp)

    # Grid of names (Python-side)
    names_arr = np.array(palette.names, dtype=object)
    grid = names_arr[idx]

    preview = preview_from_indices(idx, palette)
    return grid, idx, preview


def write_mcfunction_flat(grid: np.ndarray, 
                          out_path: str, 
                          origin: Tuple[int, int, int] = (0, 64, 0), 
                          namespace: str = "minecraft", 
                          center: bool = False):
    """
    grid: (H,W) array of block names (strings)
    origin: (x0,y0,z0) in Minecraft
    center: if True, center mosaic around origin
    """
    H, W = grid.shape
    x0, y0, z0 = origin

    o_path = Path(out_path)
    o_path.parent.mkdir(parents=True, exist_ok=True)
    o_path.touch(exist_ok=True) 

    lines = []
    for y in range(H):
        for x in range(W):
            name = grid[y, x]
            wx = x0 + x
            wz = z0 + y
            if center:
                wx = x0 + (x - W // 2)
                wz = z0 + (y - H // 2)

            # skip "air" if you ever include it
            if name == "air":
                continue

            lines.append(f"setblock {wx} {y0} {wz} {namespace}:{name}")

    o_path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    grid, idx, preview = mosaic_dither_numba("clinic.jpeg", minecraft_blocks.MINECRAFT_BLOCKS)
    write_mcfunction_flat(grid, "out/mcify.mcfunction")
    
    # preview.show("preview.png")
    # print(grid)                         