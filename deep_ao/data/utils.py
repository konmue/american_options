from typing import Callable

import numpy as np
import torch
from torch.utils.data import TensorDataset


def prepare_training_data(
    x_n: np.ndarray,
    payoff_now: np.ndarray,
    stopping_times: np.ndarray,
    stopped_paths: np.ndarray,
    payoff: Callable,
):

    # payoff at (so far) optimal stopping time; targets for the NN
    model_input = torch.tensor(np.c_[x_n, payoff_now]).float()
    targets = torch.unsqueeze(
        torch.Tensor(payoff(stopping_times, stopped_paths)), dim=1
    )

    # setting up dataloader object for training
    return TensorDataset(model_input, targets)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
