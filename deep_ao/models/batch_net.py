from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from deep_ao.models.fnn import FNN, FNNParams


class BatchLayer(nn.Module):
    def __init__(self, input_dim: int, time_dim: int, hidden_dim: int) -> None:
        super().__init__()
        """Stacked feed forward layer."""

        self.weight_matrices = nn.Parameter(torch.rand(time_dim, hidden_dim, input_dim))
        self.biases = nn.Parameter(torch.rand(time_dim, hidden_dim))

    def forward(self, x):
        x = torch.einsum("btf, tof -> bto", x, self.weight_matrices) + self.biases
        return F.relu(x)


class StoppingNets(pl.LightningModule):
    def __init__(
        self,
        n_steps: int,
        fnn_params: FNNParams,
        payoff: Callable,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()

        models = nn.ModuleDict()
        for i in range(1, n_steps):  # no decision at 0 and N
            models[f"model_{i}"] = FNN(fnn_params)

        self.n_steps = n_steps
        self.models = models
        self.payoff = payoff
        self.learning_rate = learning_rate

    def forward(self, paths):

        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        stopping_times = torch.ones(n_paths)
        payoff_at_stop = self.payoff(n_steps, paths[:, n_steps])
        stopping_probability_at_stop = torch.ones(n_paths)

        for n in np.arange(start=n_steps - 1, stop=0, step=-1):

            x_n = paths[:, n]
            payoff_now = self.payoff(n, x_n)
            model_input = torch.cat([x_n, payoff_now])
            stopping_probability = self.models[f"model_{n}"](model_input)
            stop_idx = stopping_probability >= 0.5

            stopping_times[stop_idx] = n
            payoff_at_stop[stop_idx] = torch.from_numpy(payoff_now)[stop_idx]
            stopping_probability_at_stop[stop_idx] = stopping_probability[stop_idx]

        return -1 * payoff_at_stop * stopping_probability_at_stop

    def training_step(self, batch, *args, **kwargs):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.models.parameters, lr=self.learning_rate)


class DeepOptimalStopping(pl.LightningModule):
    def __init__(
        self,
        n_steps: int,
        fnn_params: FNNParams,
        payoff: Callable,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()

        models = nn.ModuleDict()
        for i in range(1, n_steps):  # no decision at 0 and N
            models[f"model_{i}"] = FNN(fnn_params)

        self.n_steps = n_steps
        self.models = models
        self.payoff = payoff
        self.learning_rate = learning_rate

    def forward(self, paths):

        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1

        stopping_times = torch.ones(n_paths)
        payoff_at_stop = self.payoff(n_steps, paths[:, n_steps])
        stopping_probability_at_stop = torch.ones(n_paths)

        for n in np.arange(start=n_steps - 1, stop=0, step=-1):

            x_n = paths[:, n]
            payoff_now = self.payoff(n, x_n)
            model_input = torch.cat([x_n, payoff_now])
            stopping_probability = self.models[f"model_{n}"](model_input)
            stop_idx = stopping_probability >= 0.5

            stopping_times[stop_idx] = n
            payoff_at_stop[stop_idx] = torch.from_numpy(payoff_now)[stop_idx]
            stopping_probability_at_stop[stop_idx] = stopping_probability[stop_idx]

        return -1 * payoff_at_stop * stopping_probability_at_stop

    def training_step(self, batch, *args, **kwargs):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.models.parameters, lr=self.learning_rate)
