import gc
from typing import Callable

import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm


def calculate_lower_bound(
    paths: np.ndarray,
    payoff: Callable,
    models: dict,
    alpha: float = 0.05,  # confidence level for CI
):
    # Calculating the lower pricing bound (Section 3.1 from the paper)

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    payoffs_at_stop = calculate_payoffs_at_stop(paths, payoff, models, n_steps)
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


def calculate_payoffs_at_stop(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    n_steps: int,
    time: int = 0,
):
    # Calculating the lower pricing bound (Section 3.1 from the paper)

    path_steps = paths.shape[1] - 1

    payoff_at_stop = payoff_fn(n_steps, paths[:, path_steps])

    time_index = np.arange(start=n_steps - 1, stop=time - 1, step=-1)
    path_index = np.arange(len(time_index))[::-1]

    # time_index = np.arange(start=n_steps, stop=time - 1, step=-1)
    # path_index = np.arange(len(time_index))[::-1]

    for n, i in zip(time_index, path_index):

        if n == n_steps:
            continue

        x_n = paths[:, i]
        payoff_now = payoff_fn(n, x_n)

        if n != 0:
            continuation_values = (
                models[f"model_{n}"](
                    torch.from_numpy(np.c_[x_n, payoff_fn(n, x_n)]).float()
                )
                .squeeze()
                .detach()
                .numpy()
            )
        else:
            continuation_values = models[f"model_{n}"]

        idx = payoff_now >= continuation_values
        payoff_at_stop[idx] = payoff_now[idx]

    return payoff_at_stop


def calculate_upper_bound(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    path_generator: Callable,
    n_nested_paths: int = 2000,
    alpha: float = 0.05,
):

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    all_payoffs = np.zeros((n_paths, n_steps + 1))
    all_continuation_values = np.zeros((n_paths, n_steps + 1))
    all_indicators = np.zeros((n_paths, n_steps + 1))

    all_payoffs[:, -1] = payoff_fn(n_steps, paths[:, n_steps])
    all_indicators[:, -1] = 1
    # final cont. value not needed (stays 0)

    for n in tqdm(np.arange(start=n_steps - 1, stop=-1, step=-1)):

        x_n = paths[:, n]
        payoff_now = payoff_fn(n, x_n)
        all_payoffs[:, n] = payoff_now

        if n == 0:
            paths_from_here = path_generator(
                initial_value=x_n[0], n_steps=n_steps - n, n_simulations=n_nested_paths
            )
            paths_from_here = paths_from_here[:, 1:]
            continuation_value = calculate_payoffs_at_stop(
                paths_from_here, payoff_fn, models, n_steps, time=n
            ).mean()
            all_continuation_values[:, n] = continuation_value
            all_indicators[:, n] = payoff_now[0] >= models["model_0"]

        else:
            for i in range(n_paths):
                paths_from_here = path_generator(
                    initial_value=x_n[i],
                    n_steps=n_steps - n,
                    n_simulations=n_nested_paths,
                )
                paths_from_here = paths_from_here[:, 1:]
                continuation_value = calculate_payoffs_at_stop(
                    paths_from_here, payoff_fn, models, n_steps, time=n
                ).mean()
                all_continuation_values[i, n] = continuation_value

            model_continuation_values = (
                models[f"model_{n}"](
                    torch.from_numpy(np.c_[x_n, payoff_fn(n, x_n)]).float()
                )
                .squeeze()
                .detach()
                .numpy()
            )
            all_indicators[:, n] = payoff_now >= model_continuation_values

        del paths_from_here
        gc.collect()

    martingale_increments = (
        all_payoffs[:, 1:] * all_indicators[:, 1:]
        + (1 - all_indicators[:, 1:]) * all_continuation_values[:, 1:]
        - all_continuation_values[:, :-1]
    )

    martingale = np.c_[
        np.expand_dims(np.zeros(n_paths), 1), np.cumsum(martingale_increments, axis=-1)
    ]

    U_realizations = np.amax(all_payoffs - martingale, -1)
    U = U_realizations.mean()
    sigma_estimate = U_realizations.std()

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
