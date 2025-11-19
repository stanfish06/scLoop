from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE

from .data_modules import nnRegressorDataModule


class NeuralODEregressor(pl.LightningModule):
    def __init__(
        self,
        data: nnRegressorDataModule,
        t_span: torch.Tensor,
        n_hidden=64,
        n_layers=1,
        solver="rk4",
        solver_adjoint="dopri5",
        atol_adjoint=1e-4,
        rtol_adjoint=1e-4,
        lr=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.data = data
        self.t_span = t_span.to(device)
        self.input_dim = data.input_dim
        self.output_dim = data.output_dim
        self.do_validation = data.do_validation
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation_fn = nn.Tanh()
        self.solver = solver
        self.solver_adjoint = solver_adjoint
        self.atol_adjoint = (atol_adjoint,)
        self.rtol_adjoint = (rtol_adjoint,)
        self.lr = lr
        self.weight_decay = weight_decay
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
                        self.activation_fn
                        if i < len(layers_dims) - 2
                        else nn.Identity(),  # No activation on final layer
                    ),
                )
            )

        self.model = NeuralODE(
            nn.Sequential(OrderedDict(layers)),
            sensitivity="adjoint",
            solver=self.solver,
            solver_adjoint=self.solver_adjoint,
        )

    def forward(self, x, t_span):
        return self.model(x, t_span)

    def compute_loss(self, y_hat, y):
        mse_loss = F.mse_loss(y_hat, y)
        return mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.forward(x, self.t_span)
        # select last point of solution trajectory
        y_hat = y_hat[-1]
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.forward(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
        loss = self.compute_loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.forward(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
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
        return self.forward(x, self.t_span)

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

        t_eval = batch_predictions[0][0]
        all_trajectories = torch.cat([pred[1] for pred in batch_predictions], dim=1)
        final_snapshots = all_trajectories[-1, :, :]

        t_eval_np = t_eval.detach().cpu().numpy()
        trajectories_np = all_trajectories.detach().cpu().numpy()
        final_np = final_snapshots.detach().cpu().numpy()

        return (t_eval_np, trajectories_np, final_np)


def main():
    np.random.seed(42)
    x = np.random.randn(1000, 2).astype(np.float32)
    y = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1])]).astype(np.float32)

    data_module = nnRegressorDataModule(x=x, y=y, batch_size=32)
    data_module.setup("fit")

    t_span = torch.linspace(0, 1, 5)
    model = NeuralODEregressor(data=data_module, t_span=t_span, n_hidden=32, n_layers=1)
    test_x = torch.randn(4, 2)
    t_eval, y_pred = model.forward(test_x, t_span)

    # Train the model
    model.fit()


if __name__ == "__main__":
    main()
