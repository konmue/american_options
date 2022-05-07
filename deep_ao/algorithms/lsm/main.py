import itertools

import numpy as np
import pandas as pd
from tqdm import trange

from deep_ao.algorithms.lsm.features import RandomNNFeatures, ls_features, raw_features
from deep_ao.algorithms.lsm.run import run
from deep_ao.config import STRIKE, initial_values, number_assets, simulation_params

number_paths = {
    "n_train": 20000,
    "n_upper": 2000,
    "n_lower": 20000,
}

number_paths_upper = {
    "n_train": 50000,
    "n_upper": 2000,
    "n_lower": 20000,
}

feature_keys = ["base", "ls", "r-NN"]


def main():

    np.random.seed(1)

    combinations = itertools.product(number_assets, initial_values, feature_keys)
    results = []
    for n_assets, initial_value, feature_key in combinations:

        feature_dict = {
            "base": [raw_features, 0],
            "ls": [ls_features, 0],
            "r-NN": [RandomNNFeatures(input_dim=n_assets + 1), 1.0],
        }
        print(
            f"training model for d = {n_assets}, s0 = {initial_value}, feature key = {feature_key}"
        )

        feature_map, ridge_coeff = feature_dict[feature_key]

        prices = []
        for _ in trange(10):
            out = run(
                strike=STRIKE,
                n_assets=n_assets,
                initial_value=initial_value,
                number_paths=number_paths,
                simulation_params=simulation_params,
                feature_map=feature_map,
                ridge_coeff=ridge_coeff,
                upper_bound=False,
            )
            prices.append(out)
        prices = np.array(prices)

        # Run upper bound calculation only once as it is computationally expensive
        _, U = run(
            strike=STRIKE,
            n_assets=n_assets,
            initial_value=initial_value,
            number_paths=number_paths_upper,
            simulation_params=simulation_params,
            feature_map=feature_map,
            ridge_coeff=ridge_coeff,
            upper_bound=True,
            itm_only=False,
        )

        results.append(
            [
                feature_key,
                n_assets,
                initial_value,
                np.mean(prices, 0),
                np.std(prices, axis=0, ddof=1) / np.sqrt(prices.shape[0]),
                U,
            ]
        )

    results = pd.DataFrame(results)
    results.columns = ["Features", "d", "S_0", "Price", "s.e.", "U"]
    return results


if __name__ == "__main__":
    res = main()
    print(res)
