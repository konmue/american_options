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


def main():

    result_table = []
    for n_assets in number_assets:
        for initial_value in initial_values:
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
            )
            result_table.append([n_assets, initial_value, *out])

    return result_table


if __name__ == "__main__":
    results = main()
    print(results)
