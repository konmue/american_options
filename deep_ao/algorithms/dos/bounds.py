from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def calculate_payoffs_at_stop(paths, payoff_fn, models, n_steps, time=0):

    payoff_at_stop = payoff_fn(torch.tensor([n_steps]), paths[:, -1])
    stopping_times = torch.ones(paths.shape[0]) * n_steps

    time_index = np.arange(start=n_steps - 1, stop=time - 1, step=-1)
    path_index = np.arange(len(time_index))[::-1]

    for n, i in zip(time_index, path_index):
        x_n = paths[:, i]
        payoff_now = payoff_fn(torch.tensor([n]), x_n)
        if n == 0:
            stopping_probability = payoff_now >= payoff_at_stop
            if time == 0:
                print(stopping_probability.float().mean())
        else:
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


def calculate_upper_bound(
    paths,
    payoff_fn: Callable,
    models: dict,
    path_generator: Callable,
    n_nested_paths: int = 16000,  # TODO
    alpha: float = 0.05,
):

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    all_payoffs = torch.zeros((n_paths, n_steps + 1))
    all_continuation_values = torch.zeros((n_paths, n_steps + 1))
    all_indicators = torch.zeros((n_paths, n_steps + 1))

    all_payoffs[:, -1] = payoff_fn(torch.tensor([n_steps]), paths[:, n_steps])
    all_indicators[:, -1] = 1
    # final cont. value not needed (stays 0)

    for n in tqdm(np.arange(start=n_steps - 1, stop=-1, step=-1)):

        x_n = paths[:, n]
        payoff_now = payoff_fn(torch.tensor([n]), x_n)
        all_payoffs[:, n] = payoff_now

        if n == 0:
            paths_from_here = path_generator(
                initial_value=x_n[0].numpy(),
                n_steps=n_steps - n,
                n_simulations=n_nested_paths,
            )
            paths_from_here = torch.from_numpy(paths_from_here[:, 1:]).float()
            payoffs_at_stop, _ = calculate_payoffs_at_stop(
                paths_from_here, payoff_fn, models, n_steps, time=n
            )
            continuation_value = payoffs_at_stop.mean()
            all_continuation_values[:, n] = continuation_value
            stop_idx = payoff_now[0] >= continuation_value

        else:
            for i in range(n_paths):
                paths_from_here = path_generator(
                    initial_value=x_n[i].numpy(),
                    n_steps=n_steps - n,
                    n_simulations=n_nested_paths,
                )
                paths_from_here = torch.from_numpy(paths_from_here[:, 1:]).float()
                payoffs_at_stop, _ = calculate_payoffs_at_stop(
                    paths_from_here, payoff_fn, models, n_steps, time=n
                )

                continuation_value = payoffs_at_stop.mean()
                all_continuation_values[i, n] = continuation_value

            stop_idx = (
                (
                    models[f"model_{n}"](
                        torch.cat([x_n, torch.unsqueeze(payoff_now, 1)], 1)
                    )
                    >= 0.5
                )
                .detach()
                .float()
            ).squeeze()

        all_indicators[:, n] = stop_idx

    martingale_increments = (
        all_payoffs[:, 1:] * all_indicators[:, 1:]
        + (1 - all_indicators[:, 1:]) * all_continuation_values[:, 1:]
        - all_continuation_values[:, :-1]
    )

    martingale = torch.cat(
        [torch.zeros(n_paths, 1), torch.cumsum(martingale_increments, dim=-1)], dim=-1
    )

    U_realizations = torch.amax(all_payoffs - martingale, -1)
    U = U_realizations.mean().item()
    sigma_estimate = U_realizations.std().item()

    return (
        U,
        sigma_estimate,
    )
