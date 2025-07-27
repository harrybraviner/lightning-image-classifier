"""
Builds a table containing stats for each image in a dataset.
"""
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import xxhash

_stats = {
    "width": lambda img, np_img: img.width,
    "height": lambda img, np_img: img.height,
    "r_mean": lambda img, np_img: np_img[:, :, 0].mean(),
    "g_mean": lambda img, np_img: np_img[:, :, 1].mean(),
    "b_mean": lambda img, np_img: np_img[:, :, 2].mean(),
    "r_std": lambda img, np_img: np_img[:, :, 0].std(),
    "g_std": lambda img, np_img: np_img[:, :, 1].std(),
    "b_std": lambda img, np_img: np_img[:, :, 2].std(),
}

def _is_test(image_filename):
    """
    The table we generate will include whether an image is in the test set.
    We do not want this to be a thing that the model designer is deciding.
    :arg filename: The category/filename part of the path.
    """
    p_test = 0.2  # 20% of images held out as test.
    name = os.path.splitext(os.path.basename(image_filename))[0]
    x = xxhash.xxh32_intdigest(name.encode('utf-8')) % 10000
    return (x / 10000) < p_test


def build_table(dataset_path: str, show_progress: bool = False):
    # FIXME - raw-img specific to animals10. Make this more general.
    categories = os.listdir(os.path.join(dataset_path, 'raw-img'))

    stats = {n: [] for n in _stats.keys()}
    image_paths = []
    image_categories = []
    image_is_test = []

    for category in tqdm(categories, disable=not show_progress):
        # FIXME - raw-img specific to animals10. Make this more general.
        category_path = os.path.join(dataset_path, 'raw-img', category)
        images = os.listdir(category_path)
        for image in tqdm(images, disable=not show_progress, leave=False):
            image_path = os.path.join(category_path, image)
            image_paths.append(image_path)
            image_categories.append(category)
            image_is_test.append(_is_test(image))

            # Load image and convert colorspace if necessary
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Compute all stats
            img_np = np.array(img, dtype=np.float32)
            for n, f in _stats.items():
                stats[n].append(f(img, img_np))
            
    df = pd.DataFrame(
        {'path': image_paths, 'category': image_categories, 'is_test': image_is_test} | stats
    )
    df.to_csv(os.path.join(dataset_path, 'table.csv'), index=False, header=True)

    return stats

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    args = parser.parse_args()
    
    build_table(args.dataset_path, show_progress=True)
