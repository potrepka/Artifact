from common.weights import Initializer
from models.cyclegan.layers import LeakyConvBlock
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels, filters=64, init_type='normal',
                 init_scale=0.02):
        super(Discriminator, self).__init__()

        # Discriminator
        p = (1, 2, 1, 2)
        self.net = nn.Sequential(
            LeakyConvBlock(
                in_channels, filters, kernel_size=4, stride=2, padding=p),
            LeakyConvBlock(
                filters, filters * 2, kernel_size=4, stride=2, padding=p),
            LeakyConvBlock(
                filters * 2, filters * 4, kernel_size=4, stride=2, padding=p),
            LeakyConvBlock(
                filters * 4, filters * 8, kernel_size=4, stride=1, padding=p),
            LeakyConvBlock(
                filters * 8, 1, kernel_size=4, stride=1, padding=p,
                instance_norm=False, leaky_relu=False))

        # Initialize weights
        init_weights = Initializer(init_type, init_scale)
        for module in self.net.modules():
            init_weights(module)

    def forward(self, x):
        return self.net(x)
