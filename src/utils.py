import os
import random

import torch
import numpy as np
from dotenv import load_dotenv


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
