import itertools

from deep_ao.algorithms.deep_lsm.run_deep_lsm import run
from deep_ao.config import (
    ACTIVATION_FUNCTION,
    BATCH_NORM,
    BATCH_SIZE,
    INPUT_SCALING,
    STRIKE,
    USE_XAVIER_INIT,
    fc_dims_pre,
    initial_values,
    number_assets,
    simulation_params,
)
from deep_ao.data.utils import seed_everything

STEPS = 600
# STEPS = 6

number_paths = {
    "n_train": BATCH_SIZE * STEPS,
    "n_upper": 2000,
    "n_lower": 5_000_000,
}
# number_paths["n_lower"] = 100_000

pre_fnnpl_params = {
    "fc_dims_pre": fc_dims_pre,
    "input_scaling": INPUT_SCALING,
    "batch_norm": BATCH_NORM,
    "use_xavier_init": USE_XAVIER_INIT,
    "activation_function": ACTIVATION_FUNCTION,
    "loss_fn": "mse",
}

training_schedule_first = {
    int(STEPS * 0.1): 0.1,
    int(STEPS * 0.2): 0.01,
    int(STEPS * 0.3): 0.001,
    int(STEPS * 0.5): 0.0001,
}
training_schedule_others = {
    int(n_steps / 2): lr for n_steps, lr in training_schedule_first.items()
}

number_assets = [5]
initial_values = [100.0]

SEED = 0


def main():

    seed_everything(SEED)
    combinations = itertools.product(number_assets, initial_values)
    result_table = []
    for n_assets, initial_value in combinations:
        print(f"training model for d = {n_assets}, s0 = {initial_value}")
        out = run(
            strike=STRIKE,
            n_assets=n_assets,
            initial_value=initial_value,
            batch_size=BATCH_SIZE,
            number_paths=number_paths,
            simulation_params=simulation_params,
            training_schedule_first=training_schedule_first,
            training_schedule_others=training_schedule_others,
            pre_nn_params=pre_fnnpl_params,
            upper_bound=False,
        )
        result_table.append([n_assets, initial_value, *out])

    return result_table


if __name__ == "__main__":
    results = main()
    print(results)
