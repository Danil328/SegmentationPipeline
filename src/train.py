import argparse
import os
import shutil
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from loop.callbacks import Callbacks, CheckpointSaver, Logger, TensorBoard, FreezerCallback
from loop.factory import Factory
from loop.runner import Runner
from utils import read_config, set_global_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


def create_callbacks(name, dumps):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    save_dir = Path(dumps['path']) / dumps['weights'] / name
    callbacks = Callbacks(
        [
            Logger(log_dir),
            CheckpointSaver(
                metric_name=dumps['metric_name'],
                save_dir=save_dir,
                save_name='epoch_{epoch}.pth',
                num_checkpoints=4,
                mode='max'
            ),
            TensorBoard(str(log_dir)),
            FreezerCallback()
        ]
    )
    return callbacks


def main():
    args = parse_args()
    set_global_seeds(666)
    config = read_config(args.config, "TRAIN")
    config_data = read_config(args.config, "DATA")
    config_crop = read_config(args.config, "CROP")
    pprint(config)
    factory = Factory(config['train_params'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    callbacks = create_callbacks(config['train_params']['name'], config['dumps'])
    trainer = Runner(stages=config['stages'], factory=factory, callbacks=callbacks, device=device, batch_accumulation=config.get("batch_accumulation", 0))

    train_dataset = SegmentationDataset(data_folder=os.path.join(config_data['path_to_data'], f"crop_train_images_{config_crop['width']}"),
                                        phase='train',
                                        fold=config['fold'],
                                        num_classes=config_data['n_classes'],
                                        shape=(config_data['width'], config_data['height']))

    val_dataset = SegmentationDataset(data_folder=os.path.join(config_data['path_to_data'], f"crop_train_images_{config_crop['width']}"),
                                      phase='val',
                                      fold=config['fold'],
                                      num_classes=config_data['n_classes'],
                                      shape=(config_data['width'], config_data['height']))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    os.makedirs(os.path.join(config['dumps']['path'], config['dumps']['weights'], config['train_params']['name']), exist_ok=True)
    shutil.copy(args.config, os.path.join(config['dumps']['path'], config['dumps']['weights'], config['train_params']['name'], args.config.split('/')[-1]))
    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()
