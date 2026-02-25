from __future__ import annotations 
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from PIL import Image
import numpy as np
import minecraft_blocks

# -----
# Palette handling
# -----

RGB = Tuple[int, int, int]

@dataclass(frozen=True)
class Palette:
    names: List[str] # length K
    rgb: np.ndarray # shape (K, 3), float32

    @staticmethod
    def from_blocks(blocks: Dict[str, RGB]) -> "Palette":
        names = list(blocks.keys())
        rgb = np.array([blocks[n] for n in names], dtype=np.float32)
        return Palette(names=names, rgb=rgb)


# -----
# Nearest-color matching
# -----
def nearest_index_rgb(color_rgb: np.ndarray, palette_rgb: np.ndarray) -> int:
    """
    color_rgb: shape (3,), float32
    palette_rgb: shape (K,3), float32
    returns: int index of nearest palette color by squared Euclidean distance
    """
    c = color_rgb 
    p = palette_rgb
    dist2 = np.sum(p * p, axis=1) - 2.0 * (p @ c) + float(c @ c)            
    return int(np.argmin(dist2))

# -----
# Dithering
# -----
def dither_fs_serpentine(img_rgb: np.ndarray, 
                         palette: Palette, 
                         clamp: bool = True
                         ) -> np.ndarray:
    arr = img_rgb.astype(np.float32, copy=True)
    H, W, _ = arr.shape
    idx = np.zeros((H, W), dtype=np.int32)
    pal = palette.rgb # (K, 3)

    for y in range(H):
        # choose scan direction
        if y % 2 == 0:
            x_iter = range(W)              # left -> right
            dirx = 1
        else:
            x_iter = range(W-1, -1, -1)     # right -> left
            dirx = -1

        for x in x_iter:
            # clamp BEFORE matching to avoid runaway colors
            if clamp:
                arr[y, x] = np.clip(arr[y, x], 0.0, 255.0)

            old = arr[y, x]
            k = nearest_index_rgb(old, pal)
            idx[y, x] = k
            new = pal[k]
            err = old - new

            # diffuse error (mirrored when scanning right->left)
            x1 = x + dirx
            if 0 <= x1 < W:
                arr[y, x1] += err * (7/16)

            if y + 1 < H:
                # below
                arr[y+1, x] += err * (5/16)

                # below-left and below-right depend on direction
                xl = x - dirx
                xr = x + dirx
                if 0 <= xl < W:
                    arr[y+1, xl] += err * (3/16)
                if 0 <= xr < W:
                    arr[y+1, xr] += err * (1/16)

    return idx

# -----
# I/O helpers (loading, preview)
# -----
def load_image_rgb(path: str, 
                   out_size: Tuple[int, int], 
                   resize_mode: int = Image.BILINEAR
                   ) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(out_size, resize_mode)
    return np.array(img, dtype=np.float32) # (H, W, 3) float32

def indices_to_grid(idx: np.ndarray, palette: Palette) -> np.ndarray:
    names = np.array(palette.names, dtype=object)
    return names[idx]

def preview_from_indices(idx: np.ndarray, palette: Palette) -> Image.Image:
    rgb = palette.rgb.astype(np.uint8)
    preview = rgb[idx]
    return Image.fromarray(preview, mode="RGB")

# -----
# End to end convenience
# -----
def mosaic_dither(
        image_path: str,
        blocks: Dict[str, RGB],
        out_size: Tuple[int, int] = (512, 512),
        resize_mode: int = Image.BILINEAR,
        clamp: bool = True
):
    palette = Palette.from_blocks(blocks)
    img_rgb = load_image_rgb(image_path, out_size, resize_mode=resize_mode)
    idx = dither_fs_serpentine(img_rgb, palette, clamp=clamp)
    grid = indices_to_grid(idx, palette)
    preview = preview_from_indices(idx, palette)

    return grid, idx, preview

if __name__ == "__main__":
    grid, idx, preview = mosaic_dither("gym.png", minecraft_blocks.MINECRAFT_BLOCKS)
    preview.show("preview.png")
    print(grid[0, :10])