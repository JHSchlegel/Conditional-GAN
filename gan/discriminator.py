import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: int = 3,
        conv_dim: int = 64,
        image_size: int = 32,
    ):
        super(Discriminator, self).__init__()

        self.image_size = image_size

        self.label_embedding = nn.Embedding(
            num_classes, self.image_size * self.image_size
        )

        self.main = nn.Sequential(
            nn.Conv2d(channels + 1, conv_dim, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(conv_dim * 4, 1, 4, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, label):

        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape(
            [label_embed.shape[0], 1, self.image_size, self.image_size]
        )
        x = torch.cat((x, label_embed), dim=1)

        x = self.main(x)

        return x.squeeze()
