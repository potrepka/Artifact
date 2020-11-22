import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss_func = nn.MSELoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target = self.real_label
        else:
            target = self.fake_label
        target = target.expand_as(prediction)
        return self.loss_func(prediction, target)
