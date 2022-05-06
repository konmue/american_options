from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

ITM_ONLY = True


def get_features(x, payoff, use_payoff=True):

    sorted = np.sort(x, axis=-1)
    poly = np.polynomial.hermite.hermval(sorted[:, -1], np.ones(5))
    squares = sorted[:, :-1] ** 2
    cross_products = np.empty((x.shape[0], x.shape[1] - 2))
    for i in range(x.shape[1] - 2):
        cross_products[:, i] = sorted[:, i] * sorted[:, i + 1]
    prod = np.prod(x, axis=-1)

    if use_payoff:
        return np.c_[poly, sorted[:, :-1], squares, cross_products, prod, payoff]
    else:
        return np.c_[poly, sorted[:, :-1], squares, cross_products, prod]


def __train(
    paths: np.ndarray,
    payoff_fn: Callable,
    use_payoff=True,
):
    """The LSM algorithm with neural networks as in Section 2 of the paper."""

    n_steps = paths.shape[1] - 1
    models = {}

    payoff_at_stop = payoff_fn(n_steps, paths[:, -1])

    time_index = np.arange(start=n_steps, stop=0, step=-1)
    for n in time_index:

        # paths at point n
        x_n = paths[:, n]
        payoff_now = payoff_fn(n, x_n)

        if ITM_ONLY:
            which = payoff_now > 0
        else:
            which = payoff_now < np.infty

        features = get_features(x_n[which], payoff_now[which], use_payoff)
        model = LinearRegression()
        model.fit(features, payoff_at_stop[which])
        continuation_values = model.predict(features)

        # saving the model
        models[f"model_{n}"] = model

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = payoff_now[which] >= continuation_values
        payoff_at_stop[which][idx] = payoff_now[which][idx]

    models["model_0"] = payoff_at_stop.mean()

    return models, payoff_at_stop

def train(
    paths: np.ndarray,
    payoff_fn: Callable,
    arg
):
    """The LSM algorithm with neural networks as in Section 2 of the paper."""

    n_steps = paths.shape[1] - 1
    models = {}

    # stopping times s initialized to last period
    # stopping_times = (np.ones(n_paths) * n_steps).astype(int)

    payoff_at_stop = payoff_fn(n_steps, paths[:, -1])

    time_index = np.arange(start=n_steps, stop=-1, step=-1)
    for n in time_index:

        # paths at point n
        x_n = paths[:, n]
        payoff_now = payoff_fn(n, x_n)

        features = get_features(x_n, payoff_now)
        model = LinearRegression()
        model.fit(features, payoff_at_stop)
        continuation_values = model.predict(features)

        # saving the model
        models[f"model_{n}"] = model

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = payoff_now >= continuation_values
        payoff_at_stop[idx] = payoff_now[idx]

    models["model_0"] = payoff_at_stop.mean()

    return models, payoff_at_stop


def _train(
    paths: np.ndarray,
    foo: Callable,
    use_payoff=True,
):
    """The LSM algorithm with neural networks as in Section 2 of the paper."""

    n_steps = paths.shape[1] - 1
    models = {}

    def payoff_fn(x):
        return np.maximum(np.amax(x, axis=-1) - 100.0, 0)

    df = np.exp(-0.05 * 0.1)

    payoff_at_stop = payoff_fn(paths[:, -1])

    time_index = np.arange(start=n_steps, stop=0, step=-1)
    for n in time_index:

        # paths at point n
        x_n = paths[:, n]
        payoff_now = payoff_fn(x_n)

        if ITM_ONLY:
            which = payoff_now > 0
        else:
            which = payoff_now < np.infty

        features = get_features(x_n[which], payoff_now[which], use_payoff)
        model = LinearRegression()
        model.fit(features, payoff_at_stop[which] * df)
        continuation_values = model.predict(features)

        # saving the model
        models[f"model_{n}"] = model

        # updating optimal stopping time if stopping is larger than the approximated continuation value
        idx = payoff_now[which] >= continuation_values
        payoff_at_stop[which][idx] = payoff_now[which][idx]
        payoff_at_stop[~which] *= df
        payoff_at_stop[which][~idx] *= df

    models["model_0"] = payoff_at_stop.mean() * df

    return models, payoff_at_stop


def calculate_lower_bound(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    use_payoff=True,
    alpha: float = 0.05,  # confidence level for CI
):

    payoffs_at_stop = calculate_payoffs_at_stop(paths, payoff_fn, models, use_payoff)
    L = payoffs_at_stop.mean()
    sigma_estimate = payoffs_at_stop.std()

    return (
        L,
        sigma_estimate,
    )


def _calculate_payoffs_at_stop(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    n_steps=None,
    time: int = 0,
    use_payoff=True,
):

    if n_steps is None:
        n_steps = paths.shape[1] - 1

    def payoff_fn(x):
        return np.maximum(np.amax(x, axis=-1) - 100.0, 0)

    df = np.exp(-0.05 / 3)
    payoff_at_stop = payoff_fn(paths[:, -1])

    time_index = np.arange(start=n_steps - 1, stop=time - 1, step=-1)
    path_index = np.arange(len(time_index))[::-1]

    for n, i in zip(time_index, path_index):

        x_n = paths[:, i]
        payoff_now = payoff_fn(x_n)

        if n != 0:
            features = get_features(x_n, payoff_now, use_payoff)
            continuation_values = models[f"model_{n}"].predict(features)
        else:
            continuation_values = models[f"model_{n}"]
        stop_idx = payoff_now >= continuation_values
        payoff_at_stop[stop_idx] = payoff_now[stop_idx]
        payoff_at_stop[~stop_idx] *= df

    return payoff_at_stop


def __calculate_payoffs_at_stop(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
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
            features = get_features(x_n, payoff_now)
            continuation_values = models[f"model_{n}"].predict(features)
        else:
            continuation_values = models[f"model_{n}"]
        stop_idx = payoff_now >= continuation_values
        payoff_at_stop[stop_idx] = payoff_now[stop_idx]

    return


def calculate_payoffs_at_stop(
    paths: np.ndarray,
    payoff_fn: Callable,
    models: dict,
    n_steps=None,
    time: int = 0,
    use_payoff=True,
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
            features = get_features(x_n, payoff_now, use_payoff)
            continuation_values = models[f"model_{n}"].predict(features)
        else:
            continuation_values = models[f"model_{n}"]
        stop_idx = payoff_now >= continuation_values
        payoff_at_stop[stop_idx] = payoff_now[stop_idx]

    return payoff_at_stop


def calculate_upper_bound(
    paths: np.ndarray,
    payoff_fn: Callable,
    path_generator: Callable,
    models: dict,
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
        features = get_features(x_n, current_payoff)

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
            # check dim here; if cont value includes payoff now; could this be wrong?
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
