import os
import random

import torch
import numpy as np
from dotenv import load_dotenv


class Stereo2Mono(torch.nn.Module):
    def __init__(self):
        super(Stereo2Mono, self).__init__()

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 1:
            return x
        x = torch.mean(x, dim=0)
        x = torch.unsqueeze(x, dim=0)
        return x


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_wandb_token():
    load_dotenv()
    token = os.getenv('WANDB_APIS_KEY', None)
    if token is not None:
        return token
    raise RuntimeError("Environ has not wandb token")
