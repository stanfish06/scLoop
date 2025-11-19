from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_modules import nnRegressorDataModule


class MLPregressor(pl.LightningModule):
    def __init__(
        self,
        data: nnRegressorDataModule,
        n_hidden=128,
        n_layers=2,
        dropout=0.1,
        layer_norm=False,
        batch_norm=False,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.data = data
        self.input_dim = data.input_dim
        self.output_dim = data.output_dim
        self.do_validation = data.do_validation
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation_fn = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.n_hidden) if layer_norm else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(self.n_hidden) if batch_norm else nn.Identity()
        self.trainer = None

        self.save_hyperparameters("n_hidden", "n_layers")

        layers_dims = (
            [self.input_dim] + self.n_layers * [self.n_hidden] + [self.output_dim]
        )

        layers = []
        for i, (n_in, n_out) in enumerate(
            zip(layers_dims[:-1], layers_dims[1:], strict=False)
        ):
            layers.append(
                (
                    f"layer_{i}",
                    nn.Sequential(
                        nn.Linear(n_in, n_out, bias=True),
                        self.batch_norm
                        if i < len(layers_dims) - 2
                        else nn.Identity(),  # No batch norm on final layer
                        self.layer_norm
                        if i < len(layers_dims) - 2
                        else nn.Identity(),  # No layer norm on final layer
                        self.activation_fn
                        if i < len(layers_dims) - 2
                        else nn.Identity(),  # No activation on final layer
                        self.dropout
                        if i < len(layers_dims) - 2
                        else nn.Identity(),  # No dropout on final layer
                    ),
                )
            )

        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        mse_loss = F.mse_loss(y_hat, y)
        return mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.compute_loss(y_hat, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss" if self.do_validation else "train_loss",
        }

    def predict_step(self, batch, batch_idx):
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        return self.forward(x)

    # TODO: configure logging
    def fit(self, max_epochs: int = 100):
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            logger=False,
            enable_checkpointing=True,
        )
        self.trainer.fit(self, self.data)

    def predict(self):
        if self.trainer is None:
            raise RuntimeError(
                "Model must be trained first. Call fit() before predict()."
            )

        batch_predictions = self.trainer.predict(self, self.data)
        all_predictions = torch.cat(batch_predictions, dim=0)

        return all_predictions.detach().cpu().numpy()
