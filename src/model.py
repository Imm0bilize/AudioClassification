import torch
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2),
        )

        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
        )

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channels, staring_n_filters, num_classes):
        super(Classifier, self).__init__()

        n_filter = [2**i * staring_n_filters for i in range(3)]

        self.conv_1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=n_filter[0]
        )

        self.conv_2 = ConvolutionalBlock(
            in_channels=n_filter[0],
            out_channels=n_filter[1]
        )

        self.conv_3 = ConvolutionalBlock(
            in_channels=n_filter[1],
            out_channels=n_filter[2]
        )

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_filter[-1] * 8 * 250,
                      out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128,
                      out_features=16),
            nn.ReLU(),

            nn.Linear(in_features=16,
                      out_features=num_classes),

        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.dense_layer(x)
        return x
