from typing import Callable

import numpy as np

from deep_ao.algorithms.lsm.lsm import calculate_upper_bound, train
from deep_ao.data.geometric_bm import geometric_bm_generator
from deep_ao.payoffs.bermudan_max_call import bermudan_max_call


def run(
    strike: float,
    n_assets: int,
    initial_value: int,
    number_paths: dict,
    simulation_params: dict,
    feature_map: Callable,
    ridge_coeff: float = 0.0,
    upper_bound: bool = False,
):

    paths_train = geometric_bm_generator(
        number_paths["n_train"], n_assets, initial_value, **simulation_params
    )

    def payoff_fn(n, x):
        return bermudan_max_call(
            n,
            x,
            r=simulation_params["interest_rate"],
            N=simulation_params["n_steps"],
            T=simulation_params["delta_t"] * simulation_params["n_steps"],
            K=strike,
        )

    models, payoff_at_stop = train(
        paths_train, payoff_fn, feature_map=feature_map, ridge_coeff=ridge_coeff
    )
    price = payoff_at_stop.mean()

    if not upper_bound:
        return price

    def path_generator(
        initial_value: float,
        n_steps: int,
        n_simulations: int,
    ):
        params = {key: value for (key, value) in simulation_params.items()}
        params["n_steps"] = n_steps

        return geometric_bm_generator(n_simulations, n_assets, initial_value, **params)

    paths_upper = geometric_bm_generator(
        number_paths["n_upper"], n_assets, initial_value, **simulation_params
    )
    U = calculate_upper_bound(paths_upper, payoff_fn, path_generator, models)

    return price, U
