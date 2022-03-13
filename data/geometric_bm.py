import numpy as np


def geometric_bm_generator(
    n_simulations: int,
    n_steps: int,
    dim: int,
    initial_value: float,
    delta: float,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:

    delta_t = T / n_steps
    bm_increments = np.random.normal(
        scale=np.sqrt(delta_t), size=(n_simulations, n_steps, dim)
    )
    log_increments = (r - delta - ((sigma**2) / 2)) * delta_t + sigma * bm_increments
    log_increments = np.concatenate(
        (np.ones((n_simulations, 1, dim)) * np.log(initial_value), log_increments),
        axis=1,
    )

    return np.exp(np.cumsum(log_increments, axis=1))
