from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset
from wandb import config as Config

from src.utils import Stereo2Mono


class Emotion(Dataset):
    def __init__(self, paths_to_x: List[str], y: List[int], config: Config, augmentation: List[torch.nn.Module]):
        self._paths_to_x = paths_to_x
        self._y = y
        self._ds_len = len(paths_to_x)
        assert self._ds_len != 0
        assert self._ds_len == len(y)

        self._config = config

        self._preprocessing = torch.nn.Sequential(
            Stereo2Mono(),

            *augmentation,

            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._config.melspec_sample_rate,
                n_mels=self._config.melspec_n_mels,
                n_fft=self._config.melspec_n_fft,
                hop_length=self._config.melspec_hop_length,
                f_max=self._config.melspec_f_max
            ),
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=self._config.specaug_freq_mask_param
            ),
            torchaudio.transforms.TimeMasking(
                time_mask_param=self._config.specaug_time_mask_param
            )
        )
        self._eps = 1e-9

    def __len__(self):
        return self._ds_len

    def __getitem__(self, idx):
        image = torch.zeros(1, self._config.melspec_n_mels, self._config.img_padding_length)
        wav, sample_rate = torchaudio.load(self._paths_to_x[idx])

        mel_spectrogram = torch.log(self._preprocessing(wav) + self._eps)
        image[0, :, :mel_spectrogram.size(2)] = mel_spectrogram[:, :, :self._config.img_padding_length]
        return image, self._y[idx]