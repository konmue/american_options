number_assets = [5, 10]
initial_values = [90, 100, 110]

BATCH_SIZE = 8192

number_paths = {
    "n_train": 5_000_000,  # do they use 50m paths?
    "n_upper": 5_000_000,
    "n_lower": 5_000_000,
}


simulation_params = {
    "n_steps": 9,
    "interest_rate": 0.05,
    "delta": 0.1,
    "sigma": 0.2,
    "maturity": 3,  # in years,
}

STRIKE = 100

training_schedule_first = {100: 0.1, 1000: 0.01, 2000: 0.001, 3000: 0.0001}
training_schedule_others = {
    int(n_steps / 2): lr for n_steps, lr in training_schedule_first.items()
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
