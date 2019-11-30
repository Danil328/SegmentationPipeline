from collections import defaultdict
from time import sleep
from typing import Dict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from threading import Thread
import queue

from utils import batch2device

tqdm.monitor_interval = 0


class Runner:
    def __init__(self, factory, device, callbacks=None, stages: Dict[str, dict] = None, batch_accumulation: int = 0):
        self.factory = factory
        self.device = device
        self.stages = stages

        self._model = None
        self._metrics = None

        self.current_stage = None
        self.current_stage_name = None
        self.global_epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.loss = None

        self.callbacks = callbacks
        if callbacks is not None:
            self.callbacks.set_runner(self)

        # self.train_q = queue.Queue(10)
        # self.val_q = queue.Queue(10)
        self.train_loader = None
        self.val_loader = None
        self.step = 0
        self.batch_accumulation = batch_accumulation

    @property
    def model(self):
        if self._model is None:
            self._model = self.factory.make_model(device=self.device)
        return self._model

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = self.factory.make_metrics()
        return self._metrics

    # def insert_data(self, loader, is_train):
    #     for data in loader:
    #         if is_train:
    #             self.train_q.put(data, True)
    #         else:
    #             self.val_q.put(data, True)

    def fit(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        # tr_train = Thread(target=self.insert_data, args=(train_loader, True), name="train_queue", daemon=True)
        # tr_val = Thread(target=self.insert_data, args=(val_loader, False), name="val_queue", daemon=True)
        # tr_train.start()
        # tr_val.start()

        self.callbacks.on_train_begin()
        for stage_name, stage in self.stages.items():
            self.current_stage = stage
            self.current_stage_name = stage_name
            self.callbacks.on_stage_begin()

            self.optimizer = self.factory.make_optimizer(self.model, stage)
            self.scheduler = self.factory.make_scheduler(self.optimizer, stage)
            self.loss = self.factory.make_loss(stage, self.device)

            # while self.train_q.qsize() < 4:  # wait while queue doesnt have at least half of size
            #     print(f"queue is not ready, queue size - {self.train_q.qsize(), self.val_q.qsize()}")
            #     sleep(10)
            # print(f"queue is ready, queue size - {self.train_q.qsize(), self.val_q.qsize()}")

            self._run_one_stage()
            self.callbacks.on_stage_end()
            torch.cuda.empty_cache()
        self.callbacks.on_train_end()

        # tr_train.join()
        # tr_val.join()

    def _run_one_stage(self):
        for epoch in range(self.current_stage['epochs']):
            # train_loader.dataset.update_empty_mask_ratio(epoch)
            # print(f'positive ratio: {train_loader.dataset.positive_ratio}')
            self.callbacks.on_epoch_begin(self.global_epoch)

            self.model.train()
            self.metrics.train_metrics = self._run_one_epoch(epoch, self.train_loader, is_train=True)

            self.model.eval()
            self.metrics.val_metrics = self._run_one_epoch(epoch, self.val_loader, is_train=False)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.metrics.val_metrics['dice'], epoch)
            else:
                self.scheduler.step(epoch)
            self.callbacks.on_epoch_end(self.global_epoch)
            self.global_epoch += 1

    def _run_one_epoch(self, epoch: int, loader: DataLoader, is_train: bool = True) -> Dict[str, float]:
        epoch_report = defaultdict(float)
        progress_bar = tqdm(
            iterable=enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch} {['validation', 'train'][is_train]}ing...",
            ncols=0
        )
        metrics = {}
        with torch.set_grad_enabled(is_train):
            self.step = 0
            for i, data in progress_bar:
                # if is_train:
                #     while self.train_q.empty():
                #         sleep(5)
                #     data = self.train_q.get(True, 30)
                # else:
                #     while self.val_q.empty():
                #         sleep(5)
                #     data = self.val_q.get(True, 30)
                # print(f"queue is ready, queue size - {self.train_q.qsize(), self.val_q.qsize()}")

                self.callbacks.on_batch_begin(i)
                step_report = self._make_step(data, is_train)
                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value
                metrics = {k: v / (i + 1) for k, v in epoch_report.items()}
                progress_bar.set_postfix(**{k: f'{v:.5f}' for k, v in metrics.items()})
                self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)
        return metrics

    def _make_step(self, data: Dict[str, torch.Tensor], is_train: bool) -> Dict[str, float]:
        self.step += 1
        report = {}
        data = self.batch2device(data)
        images = data['image']
        masks = data['mask']

        predictions = self.model(images)
        loss = self.loss(predictions, masks)
        report['loss'] = loss.data

        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            report['grad'] = grad_norm
            if self.batch_accumulation > 0:
                if self.step % self.batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()
            for metric, f in self.metrics.functions.items():
                report['train_' + metric] = f(predictions, masks)
        else:
            for metric, f in self.metrics.functions.items():
                report[metric] = f(predictions, masks)
        return report

    def batch2device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch2device(data, device=self.device)


