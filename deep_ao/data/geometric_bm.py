import numpy as np


def geometric_bm_generator(
    n_simulations: int,
    dim: int,
    initial_value: float,
    n_steps: int,
    interest_rate: float,
    dividend_yield: float,
    sigma: float,
    maturity: float,
) -> np.ndarray:

    delta_t = maturity / n_steps
    bm_increments = np.random.normal(
        scale=np.sqrt(delta_t), size=(n_simulations, n_steps, dim)
    )
    log_increments = (
        interest_rate - dividend_yield - ((sigma**2) / 2)
    ) * delta_t + sigma * bm_increments
    log_increments = np.concatenate(
        (np.ones((n_simulations, 1, dim)) * np.log(initial_value), log_increments),
        axis=1,
    )

    return np.exp(np.cumsum(log_increments, axis=1))
