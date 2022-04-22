import gc

import numpy as np

from deep_ao.algorithms.deep_lsm.bounds import (
    calculate_lower_bound,
    calculate_upper_bound,
)
from deep_ao.algorithms.deep_lsm.deep_lsm import train
from deep_ao.config import SEED
from deep_ao.data.geometric_bm import geometric_bm_generator
from deep_ao.models.fnn import FNNLightningParams, get_dims
from deep_ao.payoffs.bermudan_max_call import bermudan_max_call


def run_deep_lsm(
    strike: int,
    n_assets: int,
    initial_value: int,
    batch_size: int,
    number_paths: dict,
    simulation_params: dict,
    training_schedule_first: dict,
    training_schedule_others: dict,
    pre_nn_params: dict,
):
    np.random.seed(SEED)
    input_dim, fc_dims = get_dims(n_assets, pre_nn_params["fc_dims_pre"])

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

    fnn_params_first = FNNLightningParams(
        input_dim,
        1,
        fc_dims,
        training_schedule_first,
        *list(pre_nn_params.values())[1:]
    )
    fnn_params_others = FNNLightningParams(
        input_dim,
        1,
        fc_dims,
        training_schedule_others,
        *list(pre_nn_params.values())[1:]
    )

    models = train(
        batch_size, paths_train, fnn_params_first, fnn_params_others, payoff_fn
    )

    # del paths_train  # free up memory
    # gc.collect()

    for name, model in models.items():
        if name[-1] == "0":
            continue
        else:
            model.eval()

    paths_lower = geometric_bm_generator(
        number_paths["n_lower"], n_assets, initial_value, **simulation_params
    )
    L, sigma_L, lower_bound = calculate_lower_bound(paths_lower, payoff_fn, models)

    # del paths_lower
    # gc.collect()

    paths_upper = geometric_bm_generator(
        number_paths["n_upper"], n_assets, initial_value, **simulation_params
    )

    def path_generator(
        initial_value: float,
        n_steps: int,
        n_simulations: int,
    ):
        params = {key: value for (key, value) in simulation_params.items()}
        params["n_steps"] = n_steps

        return geometric_bm_generator(n_simulations, n_assets, initial_value, **params)

    U, sigma_U, upper_bound = calculate_upper_bound(
        paths_upper, payoff_fn, models, path_generator
    )

    # del paths_upper
    # del models
    # gc.collect()

    summary = L, sigma_L, lower_bound, U, sigma_U, upper_bound
    print(summary)

    return summary
