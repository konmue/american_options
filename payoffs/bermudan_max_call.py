import numpy as np


def bermudan_max_call(n, x, r, N, T, K):
    discount_factor = np.exp(-r * n * T / N)
    payoff = np.maximum(np.amax(x, axis=-1) - K, 0)
    return discount_factor * payoff
