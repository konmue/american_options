from typing import Callable

import numpy as np


def calculate_payoffs_at_stop(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    feature_map: Callable,
    n_steps=None,
    time: int = 0,
):

    if n_steps is None:
        n_steps = paths.shape[1] - 1

    payoff_at_stop = payoff_fn(n_steps, paths[:, -1])

    time_index = np.arange(start=n_steps - 1, stop=time - 1, step=-1)
    path_index = np.arange(len(time_index))[::-1]

    for n, i in zip(time_index, path_index):

        x_n = paths[:, i]
        payoff_now = payoff_fn(n, x_n)

        if n != 0:
            features = feature_map(x_n, payoff_now)
            continuation_values = models[f"model_{n}"].predict(features)
        else:
            continuation_values = models[f"model_{n}"]
        stop_idx = payoff_now >= continuation_values
        payoff_at_stop[stop_idx] = payoff_now[stop_idx]

    return payoff_at_stop


def calculate_lower_bound(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    feature_map: Callable,
):

    payoffs_at_stop = calculate_payoffs_at_stop(
        paths, payoff_fn, models, feature_map=feature_map
    )
    L = payoffs_at_stop.mean()
    sigma_estimate = payoffs_at_stop.std()

    return (
        L,
        sigma_estimate,
    )


def calculate_upper_bound(
    paths: np.ndarray,
    payoff_fn: Callable,
    path_generator: Callable,
    models: dict,
    feature_map: Callable,
    n_nested_paths: int = 2000,
):

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    all_payoffs = np.zeros((n_paths, n_steps + 1))
    all_continuation_values = np.zeros((n_paths, n_steps + 1))
    all_indicators = np.zeros((n_paths, n_steps + 1))

    all_payoffs[:, -1] = payoff_fn(n_steps, paths[:, n_steps])
    all_indicators[:, -1] = 1

    time_index = np.arange(start=n_steps - 1, stop=0, step=-1)
    for n in time_index:

        x_n = paths[:, n]
        current_payoff = payoff_fn(n, x_n)
        all_payoffs[:, n] = current_payoff
        features = feature_map(x_n, current_payoff)

        for i in range(n_paths):
            paths_from_here = path_generator(
                initial_value=x_n[i],
                n_steps=n_steps - n,
                n_simulations=n_nested_paths,
            )
            paths_from_here = paths_from_here[:, 1:]

            continuation_value = calculate_payoffs_at_stop(
                paths_from_here,
                payoff_fn,
                models,
                feature_map=feature_map,
                n_steps=n_steps,
                time=n,
            ).mean()

            all_continuation_values[i, n] = continuation_value

        model_continuation_values = models[f"model_{n}"].predict(features)

        all_indicators[:, n] = current_payoff >= model_continuation_values

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

    return U
