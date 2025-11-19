import numpy as np
import pytorch_lightning as pl
from torch import Generator, from_numpy
from torch.utils.data import DataLoader, TensorDataset, random_split


class nnRegressorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_pred: np.ndarray | None = None,
        do_validation: bool = True,
        validation_fraction: float = 0.1,
        do_test: bool = True,
        test_fraction: float = 0.05,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 1,
    ):
        assert x.shape[0] == y.shape[0], (
            "x and y have different numbers of observations"
        )
        self.x = x
        self.y = y
        self.x_pred = x if x_pred is None else x_pred
        self.input_dim = self.x.shape[1]
        self.output_dim = self.y.shape[1]
        self.num_observations = self.x.shape[0]
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.do_validation = do_validation
        self.validation_fraction = validation_fraction
        self.do_test = do_test
        self.test_fraction = test_fraction
        self.data_full = TensorDataset(from_numpy(self.x), from_numpy(self.y))
        self.data_pred = TensorDataset(from_numpy(self.x_pred))
        self.allow_zero_length_dataloader_with_multiple_devices = True

    def setup(self, stage: str):
        generator = Generator().manual_seed(self.seed)
        split_fraction = [1.0]
        if self.do_validation:
            split_fraction.append(self.validation_fraction)
            split_fraction[0] = split_fraction[0] - self.validation_fraction
        else:
            split_fraction.append(0.0)
        if self.do_test:
            split_fraction.append(self.test_fraction)
            split_fraction[0] = split_fraction[0] - self.test_fraction
        else:
            split_fraction.append(0.0)
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.data_full,
            lengths=split_fraction,
            generator=generator,
        )

    def train_dataloader(self):
        generator = Generator().manual_seed(self.seed)
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            generator=generator,
        )

    def val_dataloader(self):
        return (
            []
            if not self.do_validation
            else DataLoader(self.data_val, batch_size=self.batch_size)
        )

    def test_dataloader(self):
        return (
            []
            if not self.do_test
            else DataLoader(self.data_test, batch_size=self.batch_size)
        )

    def predict_dataloader(self):
        return DataLoader(self.data_pred, batch_size=self.batch_size)

    def teardown(self, stage: str):
        return super().teardown(stage)
