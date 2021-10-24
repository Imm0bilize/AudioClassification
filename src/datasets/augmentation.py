import torchaudio
from torch import nn


class AugmentationNoise(nn.Module):
    def __init__(self, alpha=0.05):
        super(AugmentationNoise, self).__init__()

        self._alpha = alpha
        audio, _ = torchaudio.load('exercise_bike.wav')
        self._noise_audio = audio.sum(dim=0)

    def forward(self, wav):
        wav = wav + self._alpha * self._noise_audio[:wav.shape[-1]]
        wav = wav.clamp(-1, 1)
        return wav
