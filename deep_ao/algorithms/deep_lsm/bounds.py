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

    n_steps = paths.shape[1] - 1
    n_paths = paths.shape[0]

    for name, model in models.items():
        if name[-1] == "0":
            continue
        else:
            model.eval()

    for n in tqdm(np.arange(start=n_steps, stop=-1, step=-1)):

        x_n = paths[:, n]
        payoff_now = payoff(n, x_n)

        if n == n_steps:
            payoff_at_stop = payoff_now.copy()
            continue

        if n != 0:
            continuation_values = (
                models[f"model_{n}"](torch.tensor(np.c_[x_n, payoff(n, x_n)]).float())
                .squeeze()
                .detach()
                .numpy()
            )
        else:
            continuation_values = models[f"model_{n}"]

        idx = payoff_now >= continuation_values
        payoff_at_stop[idx] = payoff_now[idx]

    L = payoff_at_stop.mean()
    sigma_estimate = payoff_at_stop.std()

    return (
        L,
        sigma_estimate,
        confidence_interval_endpoint(
            upper_endpoint=False,
            bound_estimate=L,
            sigma_estimate=sigma_estimate,
            n_paths=n_paths,
            alpha=alpha,
        ),
    )


def calculate_upper_bound(
    paths: np.ndarray,
    payoff: Callable,
    models: dict,
    alpha: float = 0.05,
):

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    all_payoffs = np.zeros((n_paths, n_steps + 1))
    all_continuation_values = np.zeros((n_paths, n_steps + 1))
    all_indicators = np.zeros((n_paths, n_steps + 1))

    all_payoffs[:, -1] = payoff(n_steps, paths[:, n_steps])
    all_indicators[:, -1] = 1
    # final cont. value not needed (stays 0)

    for n in np.arange(start=n_steps - 1, stop=0, step=-1):

        x_n = paths[:, n]
        current_payoff = payoff(n, x_n)
        all_payoffs[:, n] = current_payoff

        continuation_values = (
            models[f"model_{n}"](torch.tensor(np.c_[x_n, payoff(n, x_n)]).float())
            .squeeze()
            .detach()
            .numpy()
        )

        all_continuation_values[:, n] = continuation_values
        all_indicators[:, n] = current_payoff >= continuation_values

    shifted_continuation_values = np.c_[
        np.ones((n_paths, 1)) * models["model_0"], all_continuation_values[:, 1:-1]
    ]

    martingale_increments = (
        all_payoffs[:, 1:] * all_indicators[:, 1:]
        + (1 - all_indicators[:, 1:]) * all_continuation_values[:, 1:]
        - shifted_continuation_values
    )

    martingale = np.c_[
        np.expand_dims(np.zeros(n_paths), 1), np.cumsum(martingale_increments, axis=-1)
    ]

    U_realizations = np.amax(all_payoffs - martingale, -1)
    U = U_realizations.mean()
    sigma_estimate = U_realizations.std()

    return (
        U,
        sigma_estimate,
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
