from common.weights import Initializer
from models.cyclegan.layers import ConvBlock
from models.cyclegan.layers import ConvTransposeBlock
from models.cyclegan.layers import LeakyConvBlock
from models.cyclegan.layers import ResidualBlock
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32, residual_blocks=9,
                 dropout=False, skip=False, init_type='normal',
                 init_scale=0.02):
        super(Generator, self).__init__()

        self.tanh = nn.Tanh()
        self.skip = skip

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(
                in_channels, filters, kernel_size=7, stride=1, padding=3),
            ConvBlock(
                filters, filters * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(
                filters * 2, filters * 4, kernel_size=3, stride=2, padding=1))

        # Transformer
        self.transformer = nn.Sequential(
            *[ResidualBlock(filters * 4, kernel_size=3, stride=1, padding=1,
                dropout=dropout) for _ in range(residual_blocks)])

        # Decoder
        self.decoder = nn.Sequential(
            ConvTransposeBlock(
                filters * 4, filters * 2, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvTransposeBlock(
                filters * 2, filters, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            ConvBlock(
                filters, out_channels, kernel_size=7, stride=1, padding=3,
                relu=False))

        # Generator
        self.net = nn.Sequential(self.encoder, self.transformer, self.decoder)

        # Initialize weights
        init_weights = Initializer(init_type, init_scale)
        for module in self.net.modules():
            init_weights(module)

    def forward(self, x):
        if self.skip:
            return self.tanh(self.net(x) + x)
        else:
            return self.tanh(self.net(x))
