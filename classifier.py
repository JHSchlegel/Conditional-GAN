import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Classifier, self).__init__()

        # Convolutional layers
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.0),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.0),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.0),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                ),
            ]
        )

        # Flatten
        self.flatten = nn.Flatten()

        # Linear layers
        self.linear_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(64 * 4 * 4, 64), nn.Dropout(p=0.0), nn.ReLU()),
                nn.Sequential(
                    nn.Linear(64, num_classes), nn.Dropout(p=0.0), nn.LogSoftmax(dim=1)
                ),
            ]
        )

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.flatten(x)
        for linear_block in self.linear_blocks:
            x = linear_block(x)
        return x
