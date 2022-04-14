import warnings

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


class GeometricBM:
    def __init__(
        self,
        risk_free_rate: float,
        dividend_yields: np.ndarray,
        standard_deviations: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> None:

        warnings.warn("Do not use this class; not tested yet")

        self.delta_t_factor = (
            risk_free_rate - dividend_yields - 0.5 * standard_deviations**2
        )
        self.standard_deviations = standard_deviations
        cov = np.outer(standard_deviations, standard_deviations) * correlation_matrix
        self.cholesky = np.linalg.cholesky(cov)

    def simulate(
        self,
        initial_values: np.ndarray,
        n_steps: int,
        delta_t: int,
        n_paths: int,
        include_initial_value=True,
    ):
        std_normal = np.random.normal(
            loc=0, scale=1, size=(n_paths, n_steps, len(initial_values))
        )
        correlated_normal = np.einsum("dc, bnc -> bnc", self.cholesky, std_normal)
        log_increments = (
            self.delta_t_factor * delta_t + delta_t**0.5 * correlated_normal
        )

        if include_initial_value:
            log_increments = np.concatenate(
                (
                    np.ones((n_paths, 1, len(self.standard_deviations)))
                    * np.log(initial_values),
                    log_increments,
                ),
                axis=1,
            )

        return np.exp(np.cumsum(log_increments, axis=1))
