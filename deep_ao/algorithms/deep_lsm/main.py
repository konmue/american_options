from deep_ao.algorithms.deep_lsm.run_deep_lsm import run_deep_lsm
from deep_ao.config import (
    BATCH_SIZE,
    STRIKE,
    initial_values,
    number_assets,
    number_paths,
    pre_fnnpl_params,
    simulation_params,
    training_schedule_first,
    training_schedule_others,
)


def main():

    result_table = []
    for n_assets in number_assets:
        for initial_value in initial_values:
            print(f"training model for d = {n_assets}, s0 = {initial_value}")
            L, _, lower_bound, U, _, upper_bound = run_deep_lsm(
                strike=STRIKE,
                n_assets=n_assets,
                initial_value=initial_value,
                batch_size=BATCH_SIZE,
                number_paths=number_paths,
                simulation_params=simulation_params,
                training_schedule_first=training_schedule_first,
                training_schedule_others=training_schedule_others,
                pre_nn_params=pre_fnnpl_params,
            )
            result_table.append(
                [n_assets, initial_value, L, U, lower_bound, upper_bound]
            )

    return result_table


if __name__ == "__main__":
    results = main()
    print(results)
