import gc
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from deep_ao.config import MAX_EPOCHS
from deep_ao.models.fnn import FNNLightning, FNNLightningParams


def train(
    batch_size: int,
    paths: np.ndarray,
    fnn_params_first: FNNLightningParams,  # parameters for the first network (different number of training steps)
    fnn_params_others: FNNLightningParams,  # parameters for all other networks
    payoff_fn: Callable,
):
    """The LSM algorithm with neural networks as in Section 2 of the paper."""

    n_steps = paths.shape[1] - 1
    models = {}

    payoff_at_stop = payoff_fn(n_steps, paths[:, -1])

    for n in np.arange(start=n_steps - 1, stop=0, step=-1):
        print(f"Training network {n}")

        x_n = paths[:, n]
        payoff_now = payoff_fn(n, x_n)

        model_input = torch.tensor(np.c_[x_n, payoff_now]).float()
        targets = torch.unsqueeze(torch.tensor(payoff_at_stop).float(), 1)
        dataset = TensorDataset(model_input, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

        # defining the models; copying weights from previous model
        if n == n_steps - 1:
            model = FNNLightning(fnn_params_first)
        else:
            state_dict = model.state_dict()
            model = FNNLightning(fnn_params_others)
            model.load_state_dict(state_dict)

        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, enable_model_summary=False)
        trainer.fit(model, dataloader)

        continuation_values = (
            model(torch.tensor(np.c_[x_n, payoff_fn(n, x_n)]).float())
            .detach()
            .numpy()
            .squeeze()
        )

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = payoff_now >= continuation_values
        payoff_at_stop[idx] = payoff_now[idx]

        # saving the model
        models[f"model_{n}"] = model

        del dataset
        del dataloader
        gc.collect()

    # value at time 0 (price) given by mean payoff at optimal stopping times
    models["model_0"] = payoff_at_stop.mean()

    return models
