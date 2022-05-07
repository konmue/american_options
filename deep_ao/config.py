
## Simulation Params

number_assets = [5, 10]
initial_values = [90, 100, 110]


simulation_params = {
    "n_steps": 9,
    "interest_rate": 0.05,
    "dividend_yield": 0.1,
    "sigma": 0.2,
    "delta_t": 0.3333333333333333,
}

STRIKE = 100

## Number paths & training params
BATCH_SIZE = 8192

# Params for DLSM
MAX_EPOCHS = 1  # not using the same paths twice
STEPS = 600

number_paths = {
    "n_train": BATCH_SIZE * STEPS,
    "n_upper": 2000,
    "n_lower": 5_000_000,
}

# Params for DOS
EPOCHS = 1000
EPOCH_SIZE = 1 * BATCH_SIZE
LEARNING_RATE = 0.001

## NN Params

fc_dims_pre = [50, 50]
INPUT_SCALING = True
BATCH_NORM = True
USE_XAVIER_INIT = True
ACTIVATION_FUNCTION = "relu"

# Params for DOS
pre_fnn_params = {
    "output_dim": 1,
    "fc_dims": fc_dims_pre,
    "input_scaling": INPUT_SCALING,
    "batch_norm": BATCH_NORM,
    "use_xavier_init": USE_XAVIER_INIT,
    "activation_function": ACTIVATION_FUNCTION,
    "final_activation": True,
}
