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

    payoff_at_stop = payoff_fn(np.array([n_steps]).astype(float), paths[:, -1])

    time_index = np.arange(start=n_steps, stop=0, step=-1)
    for n in time_index:

        # paths at point n
        x_n = paths[:, n]
        payoff_now = payoff_fn(np.array([n]).astype(float), x_n)

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
