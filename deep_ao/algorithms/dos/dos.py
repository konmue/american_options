from time import daylight
from typing import Callable

import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deep_ao.data.geometric_bm import geometric_bm_generator
from deep_ao.data.utils import prepare_training_data
from deep_ao.models.batch_net import StoppingNets
from deep_ao.models.fnn import FNN, FNNParams
from deep_ao.payoffs.bermudan_max_call import bermudan_max_call_torch

SEED = 1


def run(
    strike: int,
    n_assets: int,
    initial_value: int,
    batch_size: int,
    number_paths: dict,
    simulation_params: dict,
    fnn_params: FNNParams,
    learning_rate: float = 0.001,
):

    np.random.seed(SEED)

    paths = geometric_bm_generator(
        number_paths["n_train"], n_assets, initial_value, **simulation_params
    )
    paths = torch.from_numpy(paths).float()

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    def payoff(n, x):
        return bermudan_max_call_torch(
            n,
            x,
            r=simulation_params["interest_rate"],
            N=simulation_params["n_steps"],
            T=simulation_params["delta_t"] * simulation_params["n_steps"],
            K=strike,
        )

    models = nn.ModuleDict()
    optimizers = {}
    for i in range(1, n_steps):  # no decision at 0 and N
        model = FNN(fnn_params)
        models[f"model_{i}"] = model
        optimizers[f"opt_{i}"] = torch.optim.Adam(
            model.parameters(), learning_rate=learning_rate
        )

    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    stopping_times = torch.ones(n_paths)
    payoff_at_stop = payoff(n_steps, paths[:, n_steps])

    for n in np.arange(start=n_steps - 1, stop=0, step=-1):

        print(f"Training at {n}")

        x_n = paths[:, n]
        payoff_now = payoff(n, x_n)
        model_input = torch.cat([x_n, payoff_now])

        dataset = TensorDataset(model_input)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        with tqdm(enumerate(dataloader)) as tepoch:
            for i, x in tepoch:
                model = models[f"model_{n}"]
                optimizer = optimizer[f"opt_{n}"]

                model.train()
                optimizer.zero_grad()

                stopping_probability = model(x)
                stop_idx = stopping_probability >= 0.5

                # TODO: first loss or first updating payoff at stop?
                loss = -(
                    payoff_now * stopping_probability
                    + payoff_at_stop * (1 - stopping_probability)
                )

                loss.backward()
                optimizer.step()

                stopping_times[
                    i * batch_size : (i + 1) * batch_size
                ] = n * stop_idx + stopping_times[
                    i * batch_size : (i + 1) * batch_size
                ] * (
                    1 - stop_idx
                )

                payoff_at_stop[
                    i * batch_size : (i + 1) * batch_size
                ] = payoff_now * stop_idx + payoff_at_stop[
                    i * batch_size : (i + 1) * batch_size
                ] * (
                    1 - stop_idx
                )

                tepoch.set_postfix({"loss": loss.item()})

    return models, payoff_at_stop, stopping_times
