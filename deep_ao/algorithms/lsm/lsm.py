from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


def train(
    paths: np.ndarray,
    payoff_fn: Callable,
    feature_map: Callable,
    ridge_coeff: float = 0.0,
    itm_only=True,
):

    n_steps = paths.shape[1] - 1
    models = {}

    payoff_at_stop = payoff_fn(n_steps, paths[:, -1])

    time_index = np.arange(start=n_steps, stop=0, step=-1)
    for n in time_index:

        # paths at point n
        x_n = paths[:, n]
        payoff_now = payoff_fn(n, x_n)

        if itm_only:
            which = payoff_now > 0
        else:
            which = payoff_now < np.inf

        features = feature_map(x_n[which], payoff_now[which])

        if ridge_coeff > 0:
            model = Ridge(ridge_coeff)
        else:
            model = LinearRegression()

        model.fit(features, payoff_at_stop[which])
        continuation_values = model.predict(features)

        # saving the model
        models[f"model_{n}"] = model

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = payoff_now[which] >= continuation_values
        payoff_at_stop[which] = (
            idx * payoff_now[which] + (1 - idx) * payoff_at_stop[which]
        )

    models["model_0"] = payoff_at_stop.mean()

    return models, payoff_at_stop


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
    alpha: float = 0.05,  # confidence level for CI
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
    feature_key: str = "ls",
    n_nested_paths: int = 2000,
):

    feature_map = FEATURES[feature_key]
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
                n_steps,
                time=n,
                feature_key=feature_key,
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
    sigma_estimate = U_realizations.std()

    return U


def _train(paths: np.ndarray, discount_factor: float, payoff_fn: Callable):

    n_steps = paths.shape[1] - 1
    models = {}

    payoff_at_stop = payoff_fn(paths[:, -1])

    time_index = np.arange(start=n_steps - 1, stop=0, step=-1)
    for n in time_index:

        # paths at point n
        x_n = paths[:, n]
        payoff_now = payoff_fn(x_n)

        features = get_features(x_n, payoff_now)

        model = LinearRegression()
        model.fit(features, payoff_at_stop)
        continuation_values = model.predict(features)

        # saving the model
        models[f"model_{n}"] = model

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = payoff_now >= continuation_values
        payoff_at_stop[~idx] = payoff_at_stop[~idx] * discount_factor
        payoff_at_stop[idx] = payoff_now[idx]

    return models, payoff_at_stop
