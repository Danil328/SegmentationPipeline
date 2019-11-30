import argparse
import glob
import json
import os
import pydoc

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset
from data_preparation.make_masks import mapping
from metrics import dice_coef_numpy
from utils import read_config, mask2rle, CRF, optimize_trapezoid, rle2mask
import ttach as tta
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


def main():
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    transforms = tta.Compose([
                              tta.HorizontalFlip(),
                              tta.VerticalFlip(),
                              # tta.Rotate90([0,90,180,270])
                              ])

    # best_threshold, best_min_size_threshold = search_threshold(device, transforms)
    # best_threshold = [0.1]
    # best_min_size_threshold = 300

    # predict(best_threshold, best_min_size_threshold, device, transforms)


def search_threshold(device, transforms):
    val_dataset = SegmentationDataset(data_folder=os.path.join(config_data['path_to_data'], f"crop_train_images_{config_crop['width']}"),
                                      phase="val",
                                      fold=config['fold'],
                                      num_classes=config_data['n_classes'],
                                      shape=(config_data['width'], config_data['height']))

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)
    models = []
    for weight in glob.glob(os.path.join(config['weights'], config['name'], 'sgd/') + "*.pth"):
        model = pydoc.locate(config['model'])(**config['model_params'])
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        model.eval()
        if config['TTA'] == 'true':
            model = tta.SegmentationTTAWrapper(model, transforms)
        models.append(model)
    print(f"Use {len(models)} models.")
    assert len(models) > 0, "Models not loaded"

    masks, predicts, filenames = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            fnames = batch["filename"]
            images = batch["image"].to(device)
            mask = batch['mask'].cpu().numpy()
            batch_preds = np.zeros((images.size(0), mask.shape[1], config_data['width'], config_data['height']), dtype=np.float32)
            for model in models:
                batch_preds += torch.sigmoid(model(images)).cpu().numpy()
            batch_preds = batch_preds / len(models)

            predicts.append(batch_preds)
            masks.append(mask)
            filenames.append(fnames)

    predicts = np.vstack(predicts)
    masks = np.vstack(masks)
    filenames = [item for sublist in filenames for item in sublist]

    if config['use_dense_crf'] != 'true':
        print("Search threshold ...")
        thresholds = np.arange(0.1, 1.0, 0.05)
        if config['channel_threshold'] == 'true':
            best_threshold = []
            for channel in range(masks.shape[1]):
                scores = []
                for threshold in tqdm(thresholds):
                    score = dice_coef_numpy(preds=(predicts>threshold).astype(int), trues=masks, channel=channel)
                    print(f"{threshold} - {score}")
                    scores.append(score)
                best_score = np.max(scores)
                print(f"Best threshold - {thresholds[np.argmax(scores)]}, best score - {best_score}")
                print(f"Scores: {scores}")
                best_threshold.append(thresholds[np.argmax(scores)])
            print(f"Best thresholds - {best_threshold}")
        else:
            scores = []
            for threshold in tqdm(thresholds):
                score = dice_coef_numpy(preds=(predicts > threshold).astype(int), trues=masks)
                print(f"{threshold} - {score}")
                scores.append(score)
            best_score = np.max(scores)
            best_threshold = thresholds[np.argmax(scores)]
            print(f"Best threshold - {best_threshold}, best score - {best_score}")
            print(f"Scores: {scores}")
    else:
        best_threshold = 0.5

    print("Search min_size threshold ...")
    thresholds = np.arange(0, 1000, 100)
    scores = []
    for threshold in tqdm(thresholds):
        tmp = predicts.copy()
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i,j] = post_process(tmp[i,j], best_threshold, threshold, j,
                                        use_dense_crf=config['use_dense_crf'],
                                        image=cv2.imread(filenames[i]) if config['use_dense_crf']=='true' else None,
                                        use_dilations=config['use_dilations'],
                                        use_poligonization=config['use_poligonization'])
        score = dice_coef_numpy(preds=tmp, trues=masks)
        print(f"{threshold} - {score}")
        scores.append(score)
    best_score = np.max(scores)
    best_min_size_threshold = thresholds[np.argmax(scores)]
    print(f"Best min_size threshold - {best_min_size_threshold}, best score - {best_score}")
    print(f"Scores: {scores}")

    print("Search dilation ...")
    tmp = predicts.copy()
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            tmp[i,j] = post_process(tmp[i,j], best_threshold, best_min_size_threshold, j,
                                    use_dense_crf=config['use_dense_crf'],
                                    image=cv2.resize(cv2.imread(filenames[i]), (2048,2048)) if config['use_dense_crf']=='true' else None,
                                    use_dilations='true',
                                    use_poligonization=config['use_poligonization'])
    score = dice_coef_numpy(preds=tmp, trues=masks)
    print(f"Score with dilation: {score}")

    return best_threshold, best_min_size_threshold


def predict(best_threshold, min_size, device, transforms):
    print("Predict ...")
    test_dataset = SegmentationDataset(data_folder=os.path.join(config_data['path_to_data'], "test_images/"),
                                       phase='test', num_classes=config_data['n_classes'],
                                       shape=(config_data['width'], config_data['height']))

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)

    models = []
    for weight in glob.glob(os.path.join(config['weights'], config['name'], 'sgd/') + "*.pth"):
        model = pydoc.locate(config['model'])(**config['model_params'])
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        model.eval()
        if config['TTA'] == 'true':
            model = tta.SegmentationTTAWrapper(model, transforms)
        models.append(model)
    print(f"Use {len(models)} models.")
    assert len(models) > 0, "Models not loaded"

    predicts, filenames = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            fnames = batch["filename"]
            images = batch["image"].to(device)
            batch_preds = np.zeros((images.size(0), 1, config_data['width'], config_data['height']), dtype=np.float32)
            for model in models:
                batch_preds += torch.sigmoid(model(images)).cpu().numpy()
            batch_preds = batch_preds / len(models)
            predicts.append(batch_preds)
            filenames.append(fnames)

    predicts = np.vstack(predicts)
    filenames = [item for sublist in filenames for item in sublist]

    if os.path.exists(os.path.join(config_data['path_to_project'], 'tmp', 'predict_test_images_new_size')):
        shutil.rmtree(os.path.join(config_data['path_to_project'], 'tmp', 'predict_test_images_new_size'))
    os.makedirs(os.path.join(config_data['path_to_project'], 'tmp', 'predict_test_images_new_size'))

    for idx, filename in enumerate(tqdm(filenames)):
        image_shape = cv2.imread(os.path.join(config_data['path_to_data'], 'test_images', filename)).shape
        pred_img = post_process(np.squeeze(predicts[idx]), best_threshold, min_size, cls=0,
                                        use_dense_crf=config['use_dense_crf'],
                                        image=cv2.resize(cv2.imread(test_dataset.images[i]), (config_data['width'],config_data['height'])) if config['use_dense_crf']=='true' else None,
                                        use_dilations=config['use_dilations'],
                                        use_poligonization=config['use_poligonization'])

        pred_img = cv2.resize(pred_img, (image_shape[1], image_shape[0]))
        cv2.imwrite(os.path.join(config_data['path_to_project'], 'tmp', 'predict_test_images_new_size', f'{filename.replace("_hh", "").replace("jpg", "png")}'), pred_img)


def post_process(mask, threshold, min_size, cls, use_dense_crf="false", use_dilations="false", use_poligonization="false", image=None):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    if use_dense_crf  == 'true' and image is not None:
        mask = crf.dense_crf(np.array(cv2.resize(image, (config_data['height'], config_data['width']))).astype(np.uint8), mask)
    else:
        if not isinstance(threshold, list):
            mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)[1]
        elif isinstance(threshold, list):
            mask = (mask > threshold[cls]).astype(np.uint8)

    if use_dilations == 'true':
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

    if min_size > 0:
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8).copy())
        mask = np.zeros((config_data['width'], config_data['width']), np.float32)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                mask[p] = 1

    if use_poligonization == 'true':
        cnts, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # get largest five contour area
        poligon_mask = np.zeros_like(mask, dtype=np.float32)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h >= 15:
                poligon_mask[y : y + h, x : x + w] = 1.0

            # x = optimize_trapezoid(Polygon(c.squeeze()))
            # x0 = int(x[:].min())
            # y0 = int(x[4:].min())
            # w = int(x[:4].max() - x[:4].min())
            # h = int(x[4:].max() - x[4:].min())
            # if h >= 15:
            #     poligon_mask[y0:y0 + h, x0:x0 + w] = 1.0
        return poligon_mask

    return mask


if __name__ == '__main__':
    args = parse_args()
    config_data = read_config(args.config_file, "DATA")
    config_crop = read_config(args.config_file, "CROP")
    config = read_config(args.config_file, "TEST")
    inv_map = {v: k for k, v in mapping.items()}
    crf = CRF(h=config_data['height'], w=config_data['width'])
    main()