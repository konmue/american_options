import pytorch_lightning as pl
import torch
from torch import nn


class FNN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        fc_dims: list,
        output_dim: int,
        initial_batch_norm: bool,
        hidden_batch_norm: bool,
        act_fn,
        loss_fn,
        learning_rate: float,
    ) -> None:
        super().__init__()

        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        for i, fc_dim in enumerate(fc_dims):
            if i == 0:
                if initial_batch_norm:
                    layers = [
                        nn.BatchNorm1d(input_dim),
                        nn.Linear(input_dim, fc_dims[0]),
                        act_fn,
                    ]
                else:
                    layers = [nn.Linear(input_dim, fc_dims[0]), act_fn]

            else:
                layers.append(nn.Linear(fc_dims[i - 1], fc_dim))
                if hidden_batch_norm:
                    layers.append(nn.BatchNorm1d(fc_dim))
                layers.append(act_fn)

        layers.append(nn.Linear(fc_dims[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
