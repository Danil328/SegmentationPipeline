import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from albumentations import RandomCrop
from tqdm import tqdm

from utils import read_config


def parse_args():
    parser = argparse.ArgumentParser(description="Create mask for training")
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--stage", default="train", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config_file, stage="CROP")

    if args.stage == 'train':

        save_image_folder = "crop_train_images" + f"_{config['width']}"
        save_mask_folder = "crop_train_annotations" + f"_{config['width']}"

        aug = RandomCrop(height=config['width'], width=config['height'], always_apply=True)
        if os.path.exists(os.path.join(config['path_to_data'], save_image_folder)):
            shutil.rmtree(os.path.join(config['path_to_data'], save_image_folder))
            shutil.rmtree(os.path.join(config['path_to_data'], save_mask_folder))

        os.makedirs(os.path.join(config['path_to_data'], save_image_folder), exist_ok=True)
        os.makedirs(os.path.join(config['path_to_data'], save_mask_folder), exist_ok=True)

        images_hh = glob.glob(os.path.join(config['path_to_data'], 'train_images', '*hh*.jpg'))
        images_hv = glob.glob(os.path.join(config['path_to_data'], 'train_images', '*hv*.jpg'))
        masks = glob.glob(os.path.join(config['path_to_data'], 'train_annotations', '*.png'))


        cnt_width_without_interception = int(config['resize_to'] / config['width'])
        cnt_width_with_interception = int(cnt_width_without_interception + np.ceil(
            config['width'] / (config['interception'] * cnt_width_without_interception)))

        for i in tqdm(range(len(images_hh))):
            image_hh = cv2.imread(images_hh[i], cv2.IMREAD_UNCHANGED)
            image_hh = cv2.resize(image_hh, (config['resize_to'], config['resize_to']))
            image_hv = cv2.imread(images_hv[i], cv2.IMREAD_UNCHANGED)
            image_hv = cv2.resize(image_hv, (config['resize_to'], config['resize_to']))
            mask = cv2.imread(masks[i], cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (config['resize_to'], config['resize_to']))

            step = 0
            width, height = 0, 0
            for w in range(cnt_width_with_interception):
                height = 0
                for h in range(cnt_width_with_interception):
                    crop_hh = image_hh[width:width + config['width'], height:height + config['height']]
                    crop_hv = image_hv[width:width + config['width'], height:height + config['height']]
                    crop_mask = mask[width:width + config['width'], height:height + config['height']]

                    cv2.imwrite(images_hh[i].replace("train_images", save_image_folder).replace(".jpg", f"_step={step}.png"), crop_hh)
                    cv2.imwrite(images_hv[i].replace("train_images", save_image_folder).replace(".jpg", f"_step={step}.png"), crop_hv)
                    cv2.imwrite(masks[i].replace("train_annotations", save_mask_folder).replace(".png", f"_step={step}.png"), crop_mask)

                    step += 1
                    height += config['height'] - config['interception']

                width += config['width'] - config['interception']

            for k in range(config['k']):
                data = aug(image=np.concatenate([np.expand_dims(image_hh, -1), np.expand_dims(image_hv, -1)],-1), mask=mask)
                image_crop = data['image']
                crop_hh = image_crop[...,0]
                crop_hv = image_crop[...,1]
                crop_mask = data['mask']

                cv2.imwrite(images_hh[i].replace("train_images", save_image_folder).replace(".jpg", f"_step={step}.png"),crop_hh)
                cv2.imwrite(images_hv[i].replace("train_images", save_image_folder).replace(".jpg", f"_step={step}.png"),crop_hv)
                cv2.imwrite(masks[i].replace("train_annotations", save_mask_folder).replace(".png",f"_step={step}.png"),crop_mask)
                step += 1


    elif args.stage == 'test':
        save_image_folder = "crop_test_images" + f"_{config['width']}"

        if os.path.exists(os.path.join(config['path_to_data'], save_image_folder)):
            shutil.rmtree(os.path.join(config['path_to_data'], save_image_folder))

        os.makedirs(os.path.join(config['path_to_data'], save_image_folder), exist_ok=True)

        images_hh = glob.glob(os.path.join(config['path_to_data'], 'test_images', '*hh*.jpg'))
        images_hv = glob.glob(os.path.join(config['path_to_data'], 'test_images', '*hv*.jpg'))

        cnt_width_without_interception = int(config['resize_to'] / config['width'])
        cnt_width_with_interception = int(cnt_width_without_interception + np.ceil(
            config['width'] / (config['interception'] * cnt_width_without_interception)))

        for i in tqdm(range(len(images_hh))):
            image_hh = cv2.imread(images_hh[i], cv2.IMREAD_UNCHANGED)
            image_hh = cv2.resize(image_hh, (config['resize_to'], config['resize_to']))
            image_hv = cv2.imread(images_hv[i], cv2.IMREAD_UNCHANGED)
            image_hv = cv2.resize(image_hv, (config['resize_to'], config['resize_to']))

            step = 0
            width, height = 0, 0
            for w in range(cnt_width_with_interception):
                height = 0
                for h in range(cnt_width_with_interception):
                    crop_hh = image_hh[width:width + config['width'], height:height + config['height']]
                    crop_hv = image_hv[width:width + config['width'], height:height + config['height']]
                    cv2.imwrite(images_hh[i].replace("test_images", save_image_folder).replace(".jpg", f"_step={step}.png"), crop_hh)
                    cv2.imwrite(images_hv[i].replace("test_images", save_image_folder).replace(".jpg", f"_step={step}.png"), crop_hv)

                    step += 1
                    height += config['height'] - config['interception']

                width += config['width'] - config['interception']






