from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.fnn import FNN, FNNParams


def train_models(
    n_steps: int,  # number of time periods
    batch_size: int,
    sim_paths: np.ndarray,
    epoch_lr_schedule: dict,
    fnn_params: FNNParams,
    payoff: Callable,
):
    """The LSM algorithm with neural networks as in Section 2 of the paper."""

    n_paths = sim_paths.shape[0]
    models = {}

    # stopping times s initialized to last period
    stopping_times = (np.ones(n_paths) * n_steps).astype(int)

    for n in np.arange(start=n_steps - 1, stop=0, step=-1):
        print(f"Training network {n}")

        # paths at (so far) optimal stopping time
        stopped_path = sim_paths[np.arange(n_paths), stopping_times, :]

        # paths at point n
        x_n = sim_paths[:, n, :]

        # option payoff if exercise now (not necassary, but usefule additional feature)
        payoff_feature = payoff(n, x_n)

        # payoff at (so far) optimal stopping time; targets for the NN
        model_input = torch.tensor(np.c_[x_n, payoff_feature]).float()
        y = torch.unsqueeze(torch.Tensor(payoff(stopping_times, stopped_path)), dim=1)

        # setting up dataloader object for training
        dataset = TensorDataset(model_input, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

        # defining the models; copying weights from previous model
        if n == n_steps - 1:
            model = FNN(fnn_params)
        else:
            state_dict = model.state_dict()
            model = FNN(fnn_params)
            model.load_state_dict(state_dict)

        # training NN based on learning rate schedule
        if n == n_steps - 1:
            key = "first"
        else:
            key = "else"

        for learning_rate, max_epoch in epoch_lr_schedule[key]:
            model.learning_rate = learning_rate
            trainer = pl.Trainer(max_epochs=max_epoch)
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

    # value at time 0 (price) given by mean payoff at optimal stopping times
    c_0 = payoff(stopping_times, stopped_path).mean()

    return models, c_0


def calculate_lower_bound(
    n_steps: int,
    price_paths: np.ndarray,
    payoff: Callable,
    models: dict,
    c_0: np.ndarray,
    alpha: float = 0.05,  # confidence level for CI
):
    # Calculating the lower pricing bound (Section 3.1 from the paper)

    n_paths = price_paths.shape[0]

    for model in models.values():
        model.eval()

    for n in tqdm(np.arange(start=n_steps - 1, stop=-1, step=-1)):

        x_n = price_paths[np.arange(n_paths), n, :]
        y = g(np.ones(n_paths) * n, x_n)

        if n == n_steps - 1:
            g_k = y.copy()

        if n != 0:
            c = (
                models[f"model_{n}"](torch.tensor(np.c_[x_n, payoff(n, x_n)]).float())
                .squeeze()
                .detach()
                .numpy()
            )
        else:
            c = c_0

        idx = y >= c
        g_k[idx] = y[idx]

        L = g_k.mean()
        sigma_L = g_k.std()

        return L, sigma_L, lower_ci_bound(L, alpha, sigma_L, n_paths)


def lower_ci_bound(L, alpha, sigma, n_pricing):
    return L - norm.ppf(alpha / 2) * sigma / (n_pricing**0.5)
