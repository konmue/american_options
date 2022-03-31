from typing import Callable

import pytorch_lightning as pl
import torch
from torch import nn


class FNN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fc_dims: list,
        input_scaling: bool,
        batch_norm: bool,
        use_xavier_init: bool,
        activation_function: str,
        loss_fn: Callable,
    ) -> None:
        super().__init__()

        self.loss_fn = loss_fn

        if activation_function == "relu":
            act_fn = nn.ReLU()
        if activation_function == "tanh":
            act_fn = nn.Tanh()
        else:
            raise NotImplementedError

        dims = [input_dim, *fc_dims, output_dim]
        layers = []

        if input_scaling:
            layers.append(nn.BatchNorm1d(input_dim))

        for i, dim in enumerate(dims):

            if i == 0:
                continue

            if i == len(dims) - 1:
                layers.append(
                    nn.Linear(dims[i - 1], dim),
                )

            else:

                if batch_norm:
                    layers.extend(
                        [
                            nn.Linear(dims[i - 1], dim),
                            nn.BatchNorm1d(dim),
                            act_fn,
                        ]
                    )
                else:
                    layers.extend([nn.Linear(dims[i - 1], dim), act_fn])

        self.model = nn.Sequential(*layers)

        if use_xavier_init:
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
