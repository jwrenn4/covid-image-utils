import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

load_image = lambda filepath : Image.open(filepath)

def get_image_sizes(img_dir):
    arrs = [
        np.asarray(load_image(os.path.join(img_dir, img))) for img in os.listdir(img_dir)
    ]
    return np.unique([arr.shape[0] for arr in arrs]), np.unique([arr.shape[1] for arr in arrs])

def convert_image_size(image, heights, widths, num_combos = 10):
    average_size = (int(heights.mean()), int(widths.mean()))
    selected_heights = np.random.choice(heights, num_combos)
    selected_widths = np.random.choice(widths, num_combos)
    converted_images = [
        image.resize((h, w)).resize(average_size) for h in tqdm(selected_heights) for w in selected_widths
    ]
    return converted_images

def convert_and_save(img_loc, save_dir, heights, widths):
    base_fn = os.path.basename(img_loc)
    image = load_image(img_loc)
    converted_images = convert_image_size(image, heights, widths)
    for i in range(len(converted_images)):
        converted_images[i].save(os.path.join(save_dir, f'{i}_{base_fn}'))
