from typing import Callable

import numpy as np
import torch
from torch import nn


def calculate_payoffs_at_stop(paths, payoff_fn, models, n_steps, time=0):

    payoff_at_stop = payoff_fn(torch.tensor([n_steps]), paths[:, -1])
    stopping_times = torch.ones(paths.shape[0]) * n_steps

    time_index = np.arange(start=n_steps - 1, stop=time - 1, step=-1)
    path_index = np.arange(len(time_index))[::-1]

    for n, i in zip(time_index, path_index):
        if n == 0:
            continue
        x_n = paths[:, i]
        payoff_now = payoff_fn(torch.tensor([n]), x_n)
        model_input = torch.cat([x_n, torch.unsqueeze(payoff_now, 1)], dim=-1)
        model = models[f"model_{n}"]
        model.eval()
        stopping_probability = model(model_input)
        stop_idx = (stopping_probability >= 0.5).float().squeeze()
        payoff_at_stop = payoff_now * stop_idx + payoff_at_stop * (1 - stop_idx)
        stopping_times = stop_idx * n + (1 - stop_idx) * stopping_times

    return payoff_at_stop, stopping_times


def calculate_lower_bound(
    paths,
    payoff_fn: Callable,
    models: nn.ModuleDict,
    alpha: float = 0.05,  # confidence level for CI
):

    payoffs_at_stop, stopping_times = calculate_payoffs_at_stop(
        paths,
        payoff_fn,
        models,
        n_steps=paths.shape[1] - 1,
    )
    lower_bound = payoffs_at_stop.mean()
    sigma_estimate = payoffs_at_stop.std()
    mean_stopping_time = stopping_times.mean()

    return (lower_bound, sigma_estimate, mean_stopping_time)
