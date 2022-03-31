from dataclasses import dataclass
from typing import Callable

import pytorch_lightning as pl
import torch
from torch import nn


@dataclass
class FNNParams:
    input_dim: int
    output_dim: int
    fc_dims: list
    input_scaling: bool
    batch_norm: bool
    use_xavier_init: bool
    res_net: bool
    activation_function: str
    loss_fn: str


class FNN(pl.LightningModule):
    def __init__(self, params: FNNParams) -> None:
        super().__init__()

        self.res_net = params.res_net

        if params.loss_fn == "mse":
            self.loss_fn = nn.functional.mse_loss
        else:
            raise NotImplementedError

        if params.activation_function == "relu":
            act_fn = nn.ReLU()
        elif params.activation_function == "tanh":
            act_fn = nn.Tanh()
        else:
            raise NotImplementedError

        dims = [params.input_dim, *params.fc_dims, params.output_dim]
        layers = []

        if params.input_scaling:
            layers.append(nn.BatchNorm1d(params.input_dim))

        for i, dim in enumerate(dims):

            if i == 0:
                continue

            if i == len(dims) - 1:
                layers.append(
                    nn.Linear(dims[i - 1], dim),
                )

            else:

                if params.batch_norm:
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

        if params.use_xavier_init:
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x):

        if not self.res_net:
            return self.model(x)

        # assumes that last argument is the value of the payoff if exercised now
        payoff_now = x[:, -1].unsqueeze(dim=-1)
        return payoff_now + self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


