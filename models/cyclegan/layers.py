import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, relu=True):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(True) if relu else nn.Identity())

    def forward(self, x):
        return self.net(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, relu=True):
        super(ConvTransposeBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                padding=padding, output_padding=output_padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(True) if relu else nn.Identity())

    def forward(self, x):
        return self.net(x)

class LeakyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, instance_norm=True, leaky_relu=True):
        super(LeakyConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True)
                if instance_norm else nn.Identity(),
            nn.LeakyReLU(0.2, True) if leaky_relu else nn.Identity())

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0,
                 dropout=False):
        super(ResidualBlock, self).__init__()

        if dropout:
            self.net = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size, stride),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size, stride),
                nn.InstanceNorm2d(channels, affine=True))
        else:
            self.net = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size, stride),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(True),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size, stride),
                nn.InstanceNorm2d(channels, affine=True))

    def forward(self, x):
        return self.net(x) + x
