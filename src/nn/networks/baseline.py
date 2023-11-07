import warnings
from torchvision.transforms import Resize

from .base import *

warnings.filterwarnings("ignore")
torch.manual_seed(0)


class Baseline(nn.Module):
    def __init__(self, latent_dim=16, accumulated=False):
        super(Baseline, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = 1 if accumulated else 12
        self.res = Resize((self.latent_dim, self.latent_dim))
        self.sigmoid = nn.Sigmoid()
        self.in_cnn = nn.Sequential(
            CNNDownBlock(self.in_channels, 16, 3),
            CNNDownBlock(16, 32, 3),
        )
        self.linear_segment = nn.Sequential(
            nn.Linear(self.latent_dim * self.latent_dim * 32, 64),
            nn.Linear(64, self.latent_dim * self.latent_dim * 32),
        )
        self.out_cnn = nn.Sequential(
            CNNUpBlock(32, 16, 3, pad=1),
            CNNUpBlock(16, 8, 3),
            CNNUpBlock(8, 1, 3),
        )

    def parse_image(self, x):
        in_h, in_w = x.shape[2], x.shape[3]
        x = self.res(self.in_cnn(x))
        res_reversed = Resize((in_h - 4, in_w - 4))
        x_flattened = x.view(x.shape[0], -1)
        x_flattened_processed = self.linear_segment(x_flattened)
        x_reoriented = x_flattened_processed.view(
            x.shape[0], 32, self.latent_dim, self.latent_dim
        )
        out_x = res_reversed(x_reoriented)
        out_x = self.out_cnn(out_x)
        return self.sigmoid(out_x)

    def forward(self, x):
        return self.parse_image(x)


def test():
    from torchsummary import summary

    rand_data = torch.randn(32, 12, 24, 24)
    model = Baseline()
    op = model(rand_data)
    print(op.shape)

    summary(model, (12, 24, 24), device="cpu")


if __name__ == "__main__":
    test()
