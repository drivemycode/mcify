import cv2
from PIL import Image
import numpy as np

MAP_SIZE = (512, 512)
MINECRAFT_WOOL_RGB = {
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

if __name__ == "__main__":

    pixelate("room.jpeg", 8)
