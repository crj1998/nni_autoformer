import warnings

import torch.nn as nn
import pytorch_lightning as pl

import nni

# from nni.retiarii.evaluator.pytorch.lightning import Lightning, LightningModule, AccuracyWithLogits, Trainer
from nni.nas.evaluator.pytorch.lightning import Lightning, LightningModule, AccuracyWithLogits, Trainer


@nni.trace
class _SupervisedLearningModule(LightningModule):
    trainer: pl.Trainer
    def __init__(self, criterion, optimizer, scheduler, metrics, epochs, 
        warmup=0, learning_rate=0.001, weight_decay=0., betas=(0.9, 0.999), min_lr=0.00001, mixup=None
    ):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay', 'betas', 'epochs', 'warmup', 'min_lr')
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = nn.ModuleDict({name: cls() for name, cls in metrics.items()})
        self.mixup_fn = mixup

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y.argmax(dim=-1) if y.ndim==2 else y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log("val_acc", metric(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, metric in self.metrics.items():
            self.log('val_' + name, metric(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        for name, metric in self.metrics.items():
            self.log('test_' + name, metric(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            betas=self.hparams.betas, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = self.scheduler(
            optimizer, 
            warmup_epochs=self.hparams.warmup, 
            max_epochs=self.hparams.epochs, 
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr, 
            last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and self.running_mode == 'multi':
            # Don't report metric when sanity checking
            nni.report_intermediate_result(self._get_validation_metrics())

    def on_fit_end(self):
        if self.running_mode == 'multi':
            nni.report_final_result(self._get_validation_metrics())

    def _get_validation_metrics(self):
        if len(self.metrics) == 1:
            metric_name = next(iter(self.metrics))
            return self.trainer.callback_metrics['val_' + metric_name].item()
        else:
            warnings.warn('Multiple metrics without "default" is not supported by current framework.')
            return {name: self.trainer.callback_metrics['val_' + name].item() for name in self.metrics}


class Classification(Lightning):
    def __init__(self, criterion, optimizer, scheduler, mixup, 
        learning_rate, weight_decay, warmup,
        train_dataloaders, val_dataloaders, 
        **trainer_kwargs
    ):
        module = _SupervisedLearningModule(
            criterion, optimizer, scheduler, {'acc': AccuracyWithLogits}, trainer_kwargs.get("max_epochs", 500),
            warmup, learning_rate, weight_decay, mixup=mixup)
        super().__init__(module, Trainer(**trainer_kwargs), train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)