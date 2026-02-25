import cv2
from PIL import Image
import numpy as np

MAP_SIZE = (512, 512)
MINECRAFT_BLOCKS = {
    "white_wool": (233, 236, 236),
    "orange_wool": (240, 118, 19),
    "magenta_wool": (189, 68, 179),
    "light_blue_wool": (58, 175, 217),
    "yellow_wool": (248, 198, 39),
    "lime_wool": (112, 185, 25),
    "pink_wool": (237, 141, 172),
    "gray_wool": (62, 68, 71),
    "light_gray_wool": (142, 142, 134),
    "cyan_wool": (21, 137, 145),
    "purple_wool": (121, 42, 172),
    "blue_wool": (53, 57, 157),
    "brown_wool": (114, 71, 40),
    "green_wool": (84, 109, 27),
    "red_wool": (161, 39, 34),
    "black_wool": (20, 21, 25),
    "white_concrete": (207, 213, 214),
    "orange_concrete": (224, 97, 0),
    "magenta_concrete": (169, 48, 159),
    "light_blue_concrete": (36, 137, 199),
    "yellow_concrete": (241, 175, 21),
    "lime_concrete": (94, 168, 24),
    "pink_concrete": (214, 101, 143),
    "gray_concrete": (54, 57, 61),
    "light_gray_concrete": (125, 125, 115),
    "cyan_concrete": (21, 119, 136),
    "purple_concrete": (100, 32, 156),
    "blue_concrete": (44, 46, 143),
    "brown_concrete": (96, 59, 31),
    "green_concrete": (73, 91, 36),
    "red_concrete": (142, 32, 32),
    "black_concrete": (8, 10, 15),
}

def pixelate(img_path, pixel_size):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h,w = img.shape[:2]
    
    temp = cv2.resize(img, (w // pixel_size, h//pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_img = cv2.resize(temp, MAP_SIZE, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Pixelated image", pixelated_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def build_palette(blocks_dict):
    names = list(blocks_dict.keys())
    rgb = np.array([blocks_dict[n] for n in names], dtype=np.uint8)

    lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
    lab = lab.reshape(-1, 3).astype(np.int16)

    return names, lab

def match_blocks(image_path, blocks_dict, out_size=(512, 512)):
    img = Image.open(image_path).convert("RGB").resize(out_size, Image.NEAREST)
    rgb_img = np.array(img, dtype=np.uint8)

    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.int16)
    names, lab_palette = build_palette(blocks_dict)

    # for every pixel, for every block in lab palette, the 3 number 
    # vector difference
    diff = lab_img[:, :, None, :] - lab_palette[None, None, :, :]
    
    # diff has shape: [y, x, K, 3].
    # y:= height pixels
    # x:= width pixels
    # K:= lab color palette blocks: (lightness, green-red, blue-yellow). (0-100)
    dist2 = np.sum(diff * diff, axis = 3) 
    best = np.argmin(dist2, axis=2)  # axis=2 is the layer [wool1, wool2, .. woolK]

    grid = np.vectorize(lambda i: names[i])(best)
    return grid, best

def preview_from_indices(indices, blocks_dict):
    names = list(blocks_dict.keys())
    palette_rgb = np.array([blocks_dict[n] for n in names], dtype=np.uint8)
    return Image.fromarray(palette_rgb[indices])

if __name__ == "__main__":
    # grid, idx = match_blocks("gym.png", MINECRAFT_BLOCKS)
    grid, idx = match_blocks("room.jpeg", MINECRAFT_BLOCKS)

    preview_img = preview_from_indices(idx, MINECRAFT_BLOCKS)
    preview_img.show("preview.png")
