import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=10,
        num_classes=10,
        label_embed_size=5,
        channels=3,
        conv_dim=64,
    ):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.n_classes = num_classes

        self.label_embedding = nn.Embedding(self.n_classes, label_embed_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                self.z_dim + label_embed_size, conv_dim * 4, 4, 2, 0, bias=False
            ),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(conv_dim, channels, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x, label):
        x = x.reshape([x.shape[0], -1, 1, 1])
        label_embed = self.label_embedding(label)
        label_embed = label_embed.reshape([label_embed.shape[0], -1, 1, 1])

        x = torch.cat((x, label_embed), dim=1)

        return self.main(x)

    def generate(
        self,
        n_samples: int = 1_000,
        labels: torch.Tensor = None,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        if labels is None:
            labels = torch.randint(0, self.n_classes, (n_samples,))
        labels = labels.to(device)
        images = self.forward(
            torch.randn((n_samples, self.z_dim, 1, 1), dtype=torch.float).to(device),
            labels,
        )
        return images, labels
