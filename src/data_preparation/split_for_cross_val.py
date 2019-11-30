import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from utils import read_config


def parse_args():
    parser = argparse.ArgumentParser(description="Create mask for training")
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--n-folds", default=8, type=int)
    return parser.parse_args()


def get_stage(x, all_train_mask, train_masks_split):
    if x in all_train_mask:
        if x in train_masks_split:
            return 'train'
        else:
            return 'val'
    else:
        return 'holdout'

def replace_in_string(x, old, new):
    x = x.split('/')
    x[-1] = x[-1].replace(old, new)
    return '/'.join(x)

if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config_file, stage="DATA")
    config_crop = read_config(args.config_file, stage="CROP")


    masks = glob.glob(os.path.join(config['path_to_data'], 'train_annotations', '*.png'))
    train_mask, test_mask = train_test_split(masks, test_size=0.1)

    crop_mask_folder = "crop_train_annotations" + f"_{config_crop['width']}"
    crop_image_folder = 'crop_train_images' + f"_{config_crop['width']}"

    crop_masks = glob.glob(os.path.join(config['path_to_data'], crop_mask_folder, "*.png"))
    train_crop_mask = []
    for mask in train_mask:
        train_crop_mask += glob.glob(mask[:-4].replace("train_annotations", crop_mask_folder) + "*.png")

    skf = KFold(n_splits=args.n_folds)

    cv_df = pd.DataFrame()
    cv_df['images'] = crop_masks
    for i, (train_index, test_index) in enumerate(skf.split(train_mask)):
        train_mask_split = [train_mask[i] for i in train_index]
        train_crop_mask_split = []
        for mask in train_mask_split:
            train_crop_mask_split += glob.glob(mask[:-4].replace("train_annotations", crop_mask_folder) + "*.png")

        cv_df[f'fold_{i}'] = cv_df['images'].map(lambda x: get_stage(x, train_crop_mask, train_crop_mask_split))

    cv_df['images'] = cv_df['images'].map(lambda x: replace_in_string(x.replace(crop_mask_folder, crop_image_folder), "train_", "train_hh_"))
    cv_df.to_csv(os.path.join(config['path_to_data'], f'cross_val_DF_{config_crop["width"]}.csv'), index=False)
    print(cv_df.head())