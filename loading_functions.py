import pandas as pd
from PIL import Image
import pandas as pd
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

def convert_and_save(img_loc, save_dir, heights, widths, num_combos = 10):
    base_fn = os.path.basename(img_loc)
    image = load_image(img_loc)
    converted_images = convert_image_size(image, heights, widths, num_combos)
    for i in range(len(converted_images)):
        converted_images[i].save(os.path.join(save_dir, f'{i}_{base_fn}'))

def load_images_for_training(image_dir, metadata_file):
    image_files = os.listdir(image_dir)
    print(image_files)
    print(len(image_files))
    df = pd.read_csv(metadata_file)[['finding', 'filename']]

    image_files = [
        f for f in image_files if '_'.join(f.split('_')[1:]) in df.filename.tolist()
    ]
    print(image_files)
    images = [np.asarray(load_image(os.path.join(image_dir, img))) for img in image_files]
    for i in range(len(images)):
        if len(images[i].shape) > 2:
            images[i] = images[i][:,:,0]
    images = np.asarray(images)

    labels = []
    for img in image_files:
        sub_df = df[df.filename == '_'.join(img.split('_')[1:])]
        assert sub_df.shape[0] == 1
        labels.append(sub_df['finding'].iloc[0])

    return images, pd.Series(np.asarray(labels)).apply(lambda x : x == 'COVID-19').values.astype(int)
