import gc
import itertools

import pandas as pd

from deep_ao.algorithms.dos.run import run
from deep_ao.config import (
    BATCH_SIZE,
    EPOCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    STRIKE,
    initial_values,
    number_assets,
    number_paths,
    pre_fnn_params,
    simulation_params,
)
from deep_ao.data.utils import seed_everything


def main():

    seed_everything(0)
    combinations = itertools.product(number_assets, initial_values)
    results = []
    for n_assets, initial_value in combinations:
        print(f"training model for d = {n_assets}, s0 = {initial_value}")
        out = run(
            strike=STRIKE,
            n_assets=n_assets,
            initial_value=initial_value,
            epochs=EPOCHS,
            epoch_size=EPOCH_SIZE,
            batch_size=BATCH_SIZE,
            n_paths_lower=number_paths["n_lower"],
            simulation_params=simulation_params,
            pre_fnn_params=pre_fnn_params,
            learning_rate=LEARNING_RATE,
            upper_bound=True,
        )
        gc.collect()
        results.append([n_assets, initial_value, *out])

    results = pd.DataFrame(results)
    results.columns = ["d", "S_0", "L", "ci_lower", "U", "ci_upper"]
    return results


if __name__ == "__main__":
    res = main()
    print(res)
