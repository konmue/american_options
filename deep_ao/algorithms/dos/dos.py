from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deep_ao.data.geometric_bm import geometric_bm_generator
from deep_ao.models.fnn import FNN


def train(
    epochs,
    epoch_size,
    batch_size,
    initial_value: float,
    n_assets: int,
    simulation_params: dict,
    payoff_fn: Callable,
    fnn_params: dict,
    learning_rate,
):

    n_steps = simulation_params["n_steps"]

    models, optimizers = nn.ModuleDict(), {}
    for i in range(1, n_steps):
        model = FNN(**fnn_params)
        models[f"model_{i}"] = model
        optimizers[f"opt_{i}"] = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with tqdm(range(epochs)) as tepoch:
        for _ in tepoch:

            paths = geometric_bm_generator(
                n_simulations=epoch_size,
                dim=n_assets,
                initial_value=initial_value,
                **simulation_params,
            )
            paths = torch.from_numpy(paths).float()
            all_payoffs = torch.empty((paths.shape[0], paths.shape[1], 1))

            for n in range(paths.shape[1]):
                all_payoffs[:, n] = torch.unsqueeze(
                    payoff_fn(torch.tensor([n]), paths[:, n]), 1
                )

            payoff_at_stop = all_payoffs[:, -1]
            running_loss = 0
            for n in np.arange(start=n_steps - 1, stop=0, step=-1):
                model_input = torch.cat((paths[:, n], all_payoffs[:, n]), 1)

                dataset = TensorDataset(model_input)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                for i, x in enumerate(dataloader):

                    optimizers[f"opt_{n}"].zero_grad()

                    # Forward pass
                    stopping_probability = models[f"model_{n}"](x[0])
                    stop_idx = (stopping_probability >= 0.5).detach().float()

                    loss = (
                        (
                            payoff_at_stop[i * batch_size : (i + 1) * batch_size]
                            - all_payoffs[
                                i * batch_size : min((i + 1) * batch_size, epoch_size),
                                n,
                            ]
                        )
                        * stopping_probability
                    ).mean()

                    loss.backward()
                    optimizers[f"opt_{n}"].step()

                    # Updating the payoff at stop
                    payoff_at_stop[i * batch_size : (i + 1) * batch_size] = all_payoffs[
                        i * batch_size : min((i + 1) * batch_size, epoch_size), n
                    ] * stop_idx + payoff_at_stop[
                        i * batch_size : (i + 1) * batch_size
                    ] * (
                        1 - stop_idx
                    )

                    running_loss += loss.detach().item()

            tepoch.set_postfix({"loss": running_loss / ((i + 1) * (n_steps - 1))})

    return models
