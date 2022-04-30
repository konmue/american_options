import gc

import torch

from deep_ao.algorithms.dos.bounds import calculate_lower_bound, calculate_upper_bound
from deep_ao.algorithms.dos.dos import train
from deep_ao.data.geometric_bm import geometric_bm_generator
from deep_ao.payoffs.bermudan_max_call import bermudan_max_call_torch


def run(
    strike: float,
    n_assets: int,
    initial_value: float,
    epochs: int,
    epoch_size: int,
    batch_size: int,
    n_paths_lower: int,
    simulation_params: dict,
    pre_fnn_params: dict,
    learning_rate: float,
):

    fnn_params = pre_fnn_params
    fnn_params["input_dim"] = n_assets + 1

    def payoff_fn(n, x):
        return bermudan_max_call_torch(
            n,
            x,
            r=simulation_params["interest_rate"],
            N=simulation_params["n_steps"],
            T=simulation_params["delta_t"] * simulation_params["n_steps"],
            K=strike,
        )

    models = train(
        epochs,
        epoch_size,
        batch_size,
        initial_value,
        n_assets,
        simulation_params,
        payoff_fn,
        fnn_params,
        learning_rate,
    )

    gc.collect()

    paths_lower = geometric_bm_generator(
        n_simulations=n_paths_lower,
        dim=n_assets,
        initial_value=initial_value,
        **simulation_params
    )
    paths_lower = torch.from_numpy(paths_lower).float()

    (lower_bound, sigma_estimate, mean_stopping_time) = calculate_lower_bound(
        paths_lower, payoff_fn, models
    )

    def path_generator(
        initial_value: float,
        n_steps: int,
        n_simulations: int,
    ):
        params = {key: value for (key, value) in simulation_params.items()}
        params["n_steps"] = n_steps

        return geometric_bm_generator(n_simulations, n_assets, initial_value, **params)

    paths_upper = geometric_bm_generator(
        n_simulations=1024,
        dim=n_assets,
        initial_value=initial_value,
        **simulation_params
    )
    paths_upper = torch.from_numpy(paths_upper).float()

    (U, _sigma_estimate) = calculate_upper_bound(
        paths_upper, payoff_fn, models, path_generator
    )

    return [lower_bound, sigma_estimate, mean_stopping_time, U, _sigma_estimate]
