import numpy as np
import torch


def bermudan_max_call(n, x, r, N, T, K):
    discount_factor = np.exp(-r * n * T / N)
    payoff = np.maximum(np.amax(x, axis=-1) - K, 0)
    return discount_factor * payoff


def bermudan_max_call_torch(n, x, r, N, T, K):
    discount_factor = torch.exp(-r * n * T / N)
    payoff = torch.maximum(torch.amax(x, dim=-1) - K, torch.tensor([0.0]))
    return discount_factor * payoff
