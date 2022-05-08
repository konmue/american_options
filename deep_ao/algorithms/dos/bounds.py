import gc
from typing import Callable

import numpy as np
import torch
from scipy.stats import norm
from torch import nn
from tqdm import tqdm


def calculate_payoffs_at_stop(paths, payoff_fn, models, n_steps, time=0):

    payoff_at_stop = payoff_fn(torch.tensor([n_steps]), paths[:, -1])
    stopping_times = torch.ones(paths.shape[0]) * n_steps

    time_index = np.arange(start=n_steps, stop=time - 1, step=-1)
    path_index = np.arange(len(time_index))[::-1]

    for n, i in zip(time_index, path_index):

        # Always stop at final step; already accounted for in initialization
        if n == n_steps:
            continue

        x_n = paths[:, i]
        payoff_now = payoff_fn(torch.tensor([n]), x_n)

        # Always or never stop at beginning (see remark 6)
        if n == 0:
            stopping_probability = payoff_now >= payoff_at_stop

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

    n_paths = paths.shape[0]

    payoffs_at_stop, stopping_times = calculate_payoffs_at_stop(
        paths,
        payoff_fn,
        models,
        n_steps=paths.shape[1] - 1,
    )
    L = payoffs_at_stop.mean()
    sigma_estimate = payoffs_at_stop.std()

    return (
        L,
        confidence_interval_endpoint(
            upper_endpoint=False,
            bound_estimate=L,
            sigma_estimate=sigma_estimate,
            n_paths=n_paths,
            alpha=alpha,
        ),
    )


def calculate_upper_bound(
    paths,
    payoff_fn: Callable,
    models: dict,
    path_generator: Callable,
    L: float,
    n_nested_paths: int = 16000,
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
            # using the lower bound approximation for approximating  the first cont. value
            all_continuation_values[:, n] = L
            stop_idx = payoff_now[0] >= L  #  See Remark 6

        else:
            for i in range(n_paths):
                paths_from_here = path_generator(
                    initial_value=x_n[i].numpy(),
                    n_steps=n_steps - n,
                    n_simulations=n_nested_paths,
                )
                paths_from_here = torch.from_numpy(paths_from_here[:, 1:]).float()
                payoffs_at_stop, _ = calculate_payoffs_at_stop(
                    paths_from_here, payoff_fn, models, n_steps, time=n + 1
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

            del paths_from_here
            gc.collect()

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
        confidence_interval_endpoint(
            upper_endpoint=True,
            bound_estimate=U,
            sigma_estimate=sigma_estimate,
            n_paths=n_paths,
            alpha=alpha,
        ),
    )


def confidence_interval_endpoint(
    upper_endpoint: bool,
    bound_estimate: float,
    sigma_estimate: float,
    n_paths: int,
    alpha: float = 0.05,
):
    sign = 1 if upper_endpoint else -1
    return bound_estimate + sign * norm.ppf(1 - (alpha / 2)) * sigma_estimate / (
        n_paths**0.5
    )
