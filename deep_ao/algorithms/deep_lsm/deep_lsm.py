import gc
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deep_ao.algorithms.deep_lsm.config import MAX_EPOCHS
from deep_ao.data.utils import prepare_training_data
from deep_ao.models.fnn import FNN, FNNParams


def deep_lsm(
    batch_size: int,
    paths: np.ndarray,
    fnn_params_first: FNNParams,  # parameters for the first network (different number of training steps)
    fnn_params_others: FNNParams,  # parameters for all other networks
    payoff: Callable,
):
    """The LSM algorithm with neural networks as in Section 2 of the paper."""

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    models = {}

    # stopping times s initialized to last period
    stopping_times = (np.ones(n_paths) * n_steps).astype(int)

    for n in np.arange(start=n_steps - 1, stop=0, step=-1):
        print(f"Training network {n}")

        # paths at (so far) optimal stopping time
        stopped_paths = paths[np.arange(n_paths), stopping_times, :]

        # paths at point n
        x_n = paths[:, n, :]

        # option payoff if exercise now (not necessary, but usefule additional feature)
        payoff_now = payoff(n, x_n)

        dataset = prepare_training_data(
            x_n, payoff_now, stopping_times, stopped_paths, payoff
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        # defining the models; copying weights from previous model
        if n == n_steps - 1:
            model = FNN(fnn_params_first)
        else:
            state_dict = model.state_dict()
            model = FNN(fnn_params_others)
            model.load_state_dict(state_dict)

        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, enable_model_summary=False)
        trainer.fit(model, dataloader)

        # saving the model
        models[f"model_{n}"] = model

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = (
            payoff(n, x_n)
            >= model(torch.tensor(np.c_[x_n, payoff(n, x_n)]).float())
            .detach()
            .numpy()
            .squeeze()
        )
        stopping_times[idx] = n

        del dataset
        del dataloader
        gc.collect()

    # value at time 0 (price) given by mean payoff at optimal stopping times
    models["model_0"] = payoff(stopping_times, stopped_paths).mean()

    return models
