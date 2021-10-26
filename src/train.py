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
from train_engine import train_loop
from model import Classifier


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
    return all_paths, np.array(y).astype(np.float32)


def main():
    config = load_config('configs/config.yaml')
    token = get_wandb_token()

    wandb.login(key=token)
    wandb.init(project='AudioClassification', config=config, name='test', dir="../wandb")
    config = wandb.config

    set_seed(config.random_seed)

    x, y = prepare_data('datasets/dataset.csv')
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=config.val_split,
        shuffle=True,
        random_state=config.random_seed,
    )

    train_ds = Emotion(x_train, y_train, config, augmentation=[AugmentationNoise(alpha=0.1)])
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_ds = Emotion(x_val, y_val, config, augmentation=[AugmentationNoise(alpha=0.01)])
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Classifier(in_channels=1,
                       num_classes=7,
                       staring_n_filters=config.starting_n_filters)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loop(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_loader,
        len(train_ds)//config.batch_size,
        val_loader,
        len(val_ds)//config.batch_size,
        config.num_epochs,
        device
    )


if __name__ == '__main__':
    main()
