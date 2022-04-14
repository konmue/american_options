# number_assets = [5, 10]
number_assets = [5]
# initial_values = [90, 100, 110]
initial_values = [90, 100]


simulation_params = {
    "n_steps": 9,
    "interest_rate": 0.05,
    "dividend_yield": 0.1,
    "sigma": 0.2,
    "delta_t": 0.3333333333333333,
}

STRIKE = 100

MAX_EPOCHS = 1  # not using the same paths twice
BATCH_SIZE = 8192
STEPS = 300
training_schedule_first = {
    int(STEPS * 0.1): 0.1,
    int(STEPS * 0.2): 0.01,
    int(STEPS * 0.3): 0.001,
    int(STEPS * 0.5): 0.0001,
}
training_schedule_others = {
    int(n_steps / 2): lr for n_steps, lr in training_schedule_first.items()
}

number_paths = {
    "n_train": BATCH_SIZE * STEPS,
    "n_upper": 2000,
    "n_lower": 5_000_000,
}

pre_nn_params = {
    "fc_dims_pre": [50, 50],
    "input_scaling": True,
    "batch_norm": True,
    "use_xavier_init": True,
    "res_net": False,
    "activation_function": "relu",
    "loss_fn": "mse",
}
