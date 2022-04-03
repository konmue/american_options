from typing import Callable

import numpy as np
import torch

from deep_lsm import confidence_interval_endpoint


def upper_bound(
    paths: np.ndarray,
    payoff: Callable,
    models: dict,
    c_0: np.ndarray,
    alpha: float = 0.05,
):

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    all_payoffs = np.zeros((n_paths, n_steps + 1))
    all_continuation_values = np.zeros((n_paths, n_steps + 1))
    all_indicators = np.zeros((n_paths, n_steps + 1))

    all_payoffs[:, -1] = payoff(n_steps, paths[:, n_steps])
    all_indicators[:, -1] = 1

    test = np.zeros(n_steps + 1)
    test[-1] = 1
    # final cont. value not needed (stays 0)

    for n in np.arange(start=n_steps - 1, stop=0, step=-1):

        test[n] = 2

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
        np.ones((n_paths, 1)) * c_0, all_continuation_values[:, 1:-1]
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
