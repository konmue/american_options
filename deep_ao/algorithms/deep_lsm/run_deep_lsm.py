import gc

from deep_ao.algorithms.deep_lsm.bounds import (
    calculate_lower_bound,
    calculate_upper_bound,
)
from deep_ao.algorithms.deep_lsm.deep_lsm import deep_lsm
from deep_ao.data.geometric_bm import geometric_bm_generator
from deep_ao.models.fnn import FNNParams, get_dims
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
    input_dim, fc_dims = get_dims(n_assets, pre_nn_params["fc_dims_pre"])

    paths_train = geometric_bm_generator(
        number_paths["n_train"], n_assets, initial_value, **simulation_params
    )

    def payoff(n, x):
        return bermudan_max_call(
            n,
            x,
            r=simulation_params["interest_rate"],
            N=simulation_params["n_steps"],
            T=simulation_params["maturity"],
            K=strike,
        )

    fnn_params_first = FNNParams(
        input_dim,
        1,
        fc_dims,
        training_schedule_first,
        *list(pre_nn_params.values())[1:]
    )
    fnn_params_others = FNNParams(
        input_dim,
        1,
        fc_dims,
        training_schedule_others,
        *list(pre_nn_params.values())[1:]
    )

    models = deep_lsm(
        batch_size, paths_train, fnn_params_first, fnn_params_others, payoff
    )

    del paths_train  # free up memory
    gc.collect()

    paths_lower = geometric_bm_generator(
        number_paths["n_lower"], n_assets, initial_value, **simulation_params
    )
    L, sigma_L, lower_bound = calculate_lower_bound(paths_lower, payoff, models)

    del paths_lower
    gc.collect()

    paths_upper = geometric_bm_generator(
        number_paths["n_upper"], n_assets, initial_value, **simulation_params
    )
    U, sigma_U, upper_bound = calculate_upper_bound(paths_upper, payoff, models)

    del paths_upper
    gc.collect()

    summary = L, sigma_L, lower_bound, U, sigma_U, upper_bound
    print(summary)

    return summary
