import pandas as pd
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import skimage as sk
import os

load_image = lambda filepath : Image.open(filepath)

def get_image_sizes(img_dir):
    arrs = [
        np.asarray(load_image(os.path.join(img_dir, img))) for img in os.listdir(img_dir)
    ]
    return np.unique([arr.shape[0] for arr in arrs]), np.unique([arr.shape[1] for arr in arrs])

def convert_image_size(image, heights, widths, num_combos = 10, final_size = (64, 64)):
    selected_heights = np.random.choice(heights, num_combos)
    selected_widths = np.random.choice(widths, num_combos)
    converted_images = [
        image.resize((h, w)).resize(final_size) for h in tqdm(selected_heights) for w in selected_widths
    ]
    return converted_images

def convert_and_save(img_loc, save_dir, heights, widths, num_combos = 10):
    base_fn = os.path.basename(img_loc)
    image = load_image(img_loc)
    converted_images = convert_image_size(image, heights, widths, num_combos)
    for i in range(len(converted_images)):
        random_degree_rotation = (np.random.random() - 0.5) * 50 # between -25 and 25 degrees
        to_rotate = np.random.random() > 0.5
        to_noise = np.random.random() > 0.5
        to_flip = np.random.random() > 0.5
        img = np.asarray(converted_images[i])
        if to_rotate:
            img = sk.transform.rotate(img, random_degree_rotation)
        if to_noise:
            img = sk.util.random_noise(img)
        if to_flip:
            img = img[:, ::-1]
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f'{i}_{base_fn}'))

def load_images_for_training(image_dir, metadata_file, num_training = 0.5):
    image_files = os.listdir(image_dir)
    df = pd.read_csv(metadata_file)[['finding','filename']]
    image_data = {}
    
    # extract data in dictionary to keep file names separate
    for fn in tqdm(df.filename.unique(), desc = 'Aggregation:'):
        this_image_data = {
            'images' : [],
            'labels' : []
        }
        label = 1 if df[df.filename == fn].finding.iloc[0] == 'COVID-19' else 0
        files_of_name = [f for f in image_files if fn in f]
        for f in files_of_name:
            img_arr = np.asarray(load_image(os.path.join(image_dir, f)))
            if len(img_arr.shape) > 2:
                img_arr = img_arr[:,:,0]
            img_arr = (img_arr / img_arr.max()).reshape((img_arr.shape[0], img_arr.shape[1], 1)).astype(list)
            this_image_data['images'].append(img_arr)
            this_image_data['labels'].append(label)
        image_data[fn] = this_image_data
    
    # now separate into training and testing
    if isinstance(num_training, float):
        if not ((num_training > 0) and (num_training < 1)):
            raise ValueError('If float-valued, num_training must be between 0 and 1')
        num_training = int(len(list(image_data.keys()))*num_training)
    train_keys = np.random.choice(list(image_data.keys()), num_training, replace = False)
    test_keys = [k for k in image_data.keys() if k not in train_keys]
    x_train, x_test, y_train, y_test = [], [], [], []
    for k in tqdm(train_keys, desc = 'Creating Training'):
        x_train.extend(image_data[k]['images'])
        y_train.extend(image_data[k]['labels'])
    for k in tqdm(test_keys, desc = 'Creating Testing'):
        x_test.extend(image_data[k]['images'])
        y_test.extend(image_data[k]['labels'])
    
    return np.asarray(x_train).astype(float), np.asarray(x_test).astype(float), np.asarray(y_train).reshape(-1, 1).astype(float), np.asarray(y_test).reshape(-1, 1).astype(float)
    
