import yaml

import wandb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import get_wandb_token, set_seed
from datasets.dataset import Emotion
from datasets.augmentation import AugmentationNoise


def load_config(path):
    try:
        with open(path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except OSError:
        raise RuntimeError(f'Config file on path:{path} not found')
    return config


def load_dataframe(path):
    try:
        df = pd.read_csv(path, sep=';')
    except OSError:
        raise RuntimeError(
            f'Dataframe with dataset on path:{path} not found. Change path or create it with dataset/dataset_sheet.py'
        )
    return df


def prepare_data(path):
    df = load_dataframe(path)
    all_paths = np.array(df['path'].tolist())
    all_class_numbers = df['class_number'].tolist()
    assert len(all_paths) == len(all_class_numbers)

    num_classes = len(df['class_number'].unique())

    y = []
    for class_number in all_class_numbers:
        tmp = np.zeros(num_classes)
        tmp[class_number-1] = 1.0
        y.append(tmp)
    return all_paths, y


def main():
    config = load_config('configs/config.yaml')
    token = get_wandb_token()

    wandb.login(key=token)
    wandb.init(project='AudioClassification', config=config, name='test', dir="../wandb")
    config = wandb.config

    set_seed(config.random_seed)

    x, y = prepare_data('datasets/dataset.csv')
    (x_train, y_train), (x_val, y_val) = train_test_split(
        x,
        y,
        test_size=config.val_split,
        shuffle=True,
        random_state=config.random_seed,
    )

    train_ds = Emotion(x_train, y_train, config, augmentation=[AugmentationNoise(alpha=0.1)])
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=True,
    )

    val_ds = Emotion(x_val, y_val, config, augmentation=[AugmentationNoise(alpha=0.01)])
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
