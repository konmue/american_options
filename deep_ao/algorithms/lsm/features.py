import numpy as np


def ls_features(x, payoff, include_payoff=True):

    sorted_arr = np.sort(x, axis=-1)
    poly = np.polynomial.hermite.hermval(sorted_arr[:, -1], np.ones(5))
    squares = sorted_arr[:, :-1] ** 2
    cross_products = np.empty((x.shape[0], x.shape[1] - 2))
    for i in range(x.shape[1] - 2):
        cross_products[:, i] = sorted_arr[:, i] * sorted_arr[:, i + 1]
    prod = np.prod(x, axis=-1)

    if include_payoff:
        return np.c_[poly, sorted_arr[:, :-1], squares, cross_products, prod, payoff]
    else:
        return np.c_[poly, sorted_arr[:, :-1], squares, cross_products, prod]


def raw_features(x, payoff):
    return np.c_[x, payoff]


def nn_features(x, payoff, weight_matrix, bias, leaky_relu_alpha=0.0001):

    nn_input = np.c_[x, payoff]
    out = bias + np.einsum("bd, hd -> bh", nn_input, weight_matrix)

    return np.where(out > 0, out, leaky_relu_alpha * out)


class RandomNNFeatures:
    def __init__(self, input_dim=6, hidden_dim=50) -> None:
        self.weight_matrix_1 = np.random.normal(size=(hidden_dim, input_dim))
        self.bias_1 = np.random.normal(size=(hidden_dim))

    def __call__(self, x, payoff):
        out = nn_features(x, payoff, self.weight_matrix_1, self.bias_1)
        return out


