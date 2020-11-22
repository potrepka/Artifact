import torch.nn as nn

class Initializer:
    def __init__(self, init_type, init_scale):
        self.init_type = init_type
        self.init_scale = init_scale

    def __call__(self, module):
        if type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            if self.init_type == 'normal':
                nn.init.normal_(module.weight, std=self.init_scale)
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(module.weight, gain=self.init_scale)
            elif self.init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
            elif self.init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=self.init_scale)
            else:
                raise ValueError("Unknown initialization method: [%s]"
                    % self.init_type)
            nn.init.constant_(module.bias, 0)
        elif type(module) == nn.InstanceNorm2d:
            nn.init.normal_(module.weight, 1, self.init_scale)
            nn.init.constant_(module.bias, 0)
