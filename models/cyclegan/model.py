from argparse import Namespace
from collections import OrderedDict
from common.loss import GANLoss
from common.pool import Pool
import itertools
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.generator import Generator
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pytorch_lightning as pl
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

class CycleGAN(pl.LightningModule):
    def __init__(self, train_loader, val_loader, batch_size=8, precision=16,
                 iterations=0, in_channels=3, out_channels=3, g_filters=32,
                 d_filters=64, residual_blocks=9, dropout=False, skip=False,
                 learning_rate=0.0002, beta_1=0.5, beta_2=0.999,
                 init_type='normal', init_scale=0.02, pool_size_a=50,
                 pool_size_b=50, lambda_dis_a=1.0, lambda_dis_b=1.0,
                 lambda_gen_a=1.0, lambda_gen_b=1.0, lambda_cycle_a=10.0,
                 lambda_cycle_b=10.0, lambda_id_a=5.0, lambda_id_b=5.0,
                 shuffle=False):
        super(CycleGAN, self).__init__()

        if in_channels != out_channels and (lambda_id_a > 0.0 or
                                            lambda_id_b > 0.0):
            raise ValueError('Dimensions must match if using identity loss.')

        self.save_hyperparameters(
            'batch_size',
            'precision',
            'iterations',
            'in_channels',
            'out_channels',
            'g_filters',
            'd_filters',
            'residual_blocks',
            'dropout',
            'skip',
            'learning_rate',
            'beta_1',
            'beta_2',
            'init_type',
            'init_scale',
            'pool_size_a',
            'pool_size_b',
            'lambda_dis_a',
            'lambda_dis_b',
            'lambda_gen_a',
            'lambda_gen_b',
            'lambda_cycle_a',
            'lambda_cycle_b',
            'lambda_id_a',
            'lambda_id_b',
            'shuffle'
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g_filters = g_filters
        self.d_filters = d_filters
        self.residual_blocks = residual_blocks
        self.dropout = dropout
        self.skip = skip
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_dis_a = lambda_dis_a
        self.lambda_dis_b = lambda_dis_b
        self.lambda_gen_a = lambda_gen_a
        self.lambda_gen_b = lambda_gen_b
        self.lambda_cycle_a = lambda_cycle_a
        self.lambda_cycle_b = lambda_cycle_b
        self.lambda_id_a = lambda_id_a
        self.lambda_id_b = lambda_id_b

        # A -> B
        self.gen_a_to_b = Generator(in_channels, out_channels, g_filters,
                                    residual_blocks, dropout, skip, init_type,
                                    init_scale)
        # B -> A
        self.gen_b_to_a = Generator(out_channels, in_channels, g_filters,
                                    residual_blocks, dropout, skip, init_type,
                                    init_scale)

        # Data Pools
        self.fake_a_pool = Pool(pool_size_a)
        self.fake_b_pool = Pool(pool_size_b)

        # A -> real/fake
        self.dis_a = Discriminator(in_channels, d_filters, init_type,
                                   init_scale)
        # B -> real/fake
        self.dis_b = Discriminator(out_channels, d_filters, init_type,
                                   init_scale)

        # Loss Functions
        self.loss_func_gan = GANLoss()
        self.loss_func_cycle = nn.L1Loss()
        self.loss_func_id = nn.L1Loss()

    def forward(self, input):
        image_a, image_b = input
        self.real_a = image_a
        self.real_b = image_b
        self.fake_a = self.gen_b_to_a(self.real_b)
        self.fake_b = self.gen_a_to_b(self.real_a)
        self.cycle_a = self.gen_b_to_a(self.fake_b)
        self.cycle_b = self.gen_a_to_b(self.fake_a)
        if self.lambda_id_b > 0.0:
            self.id_a = self.gen_b_to_a(self.real_a)
            self.id_b = self.gen_a_to_b(self.real_b)

    def backward_gen_a_to_b(self, log=True):
        loss_gen_b = self.loss_func_gan(self.dis_b(self.fake_b), True)
        loss_gen_b *= self.lambda_gen_b
        loss_cycle_b = self.loss_func_cycle(self.cycle_b, self.real_b)
        loss_cycle_b *= self.lambda_cycle_b
        if self.lambda_id_b > 0.0:
            loss_id_b = self.loss_func_id(self.id_b, self.real_b)
            loss_id_b *= self.lambda_id_b
        else:
            loss_id_b = 0.0
        loss_gen_a_to_b = loss_gen_b + loss_cycle_b + loss_id_b
        if log:
            self.log('loss_gen_a_to_b', loss_gen_a_to_b, prog_bar=True)
            self.log('loss_gen_b', loss_gen_b)
            self.log('loss_cycle_b', loss_cycle_b)
            if self.lambda_id_b > 0.0:
                self.log('loss_id_b', loss_id_b)
        return loss_gen_a_to_b

    def backward_gen_b_to_a(self, log=True):
        loss_gen_a = self.loss_func_gan(self.dis_a(self.fake_a), True)
        loss_gen_a *= self.lambda_gen_a
        loss_cycle_a = self.loss_func_cycle(self.cycle_a, self.real_a)
        loss_cycle_a *= self.lambda_cycle_a
        if self.lambda_id_a > 0.0:
            loss_id_a = self.loss_func_id(self.id_a, self.real_a)
            loss_id_a *= self.lambda_id_a
        else:
            loss_id_a = 0.0
        loss_gen_b_to_a = loss_gen_a + loss_cycle_a + loss_id_a
        if log:
            self.log('loss_gen_b_to_a', loss_gen_b_to_a, prog_bar=True)
            self.log('loss_gen_a', loss_gen_a)
            self.log('loss_cycle_a', loss_cycle_a)
            if self.lambda_id_a > 0.0:
                self.log('loss_id_a', loss_id_a)
        return loss_gen_b_to_a

    def backward_dis_a(self, use_pool=True, log=True):
        real = self.real_a
        fake = self.fake_a
        if use_pool:
            fake = self.fake_a_pool.query(fake).detach()
        loss = self.loss_func_gan(self.dis_a(real), True)
        loss += self.loss_func_gan(self.dis_a(fake), False)
        loss *= self.lambda_dis_a
        if log:
            self.log('loss_dis_a', loss, prog_bar=True)
        return loss

    def backward_dis_b(self, use_pool=True, log=True):
        real = self.real_b
        fake = self.fake_b
        if use_pool:
            fake = self.fake_b_pool.query(fake).detach()
        loss = self.loss_func_gan(self.dis_b(real), True)
        loss += self.loss_func_gan(self.dis_b(fake), False)
        loss *= self.lambda_dis_b
        if log:
            self.log('loss_dis_b', loss, prog_bar=True)
        return loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        optimizer_gen = optim.Adam(
            itertools.chain(
                self.gen_a_to_b.parameters(),
                self.gen_b_to_a.parameters()),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2))
        optimizer_dis = optim.Adam(
            itertools.chain(
                self.dis_a.parameters(),
                self.dis_b.parameters()),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2))
        return [optimizer_gen, optimizer_dis]

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            self.forward(batch)
            loss = self.backward_gen_a_to_b()
            loss += self.backward_gen_b_to_a()
        elif optimizer_idx == 1:
            loss = self.backward_dis_a()
            loss += self.backward_dis_b()
        return loss

    def validation_step(self, batch, batch_idx):
        self.forward(batch)
        losses = [self.backward_gen_a_to_b(log=False),
                  self.backward_dis_b(use_pool=False, log=False),
                  self.backward_gen_b_to_a(log=False),
                  self.backward_dis_a(use_pool=False, log=False)]
        val_loss = torch.stack(losses).mean()
        return OrderedDict({
            'val_loss': val_loss
        })

    def validation_epoch_end(self, outputs):
        val_losses = [info['val_loss'] for info in outputs]
        avg_val_loss = torch.stack(val_losses).mean()
        self.log('val_loss', avg_val_loss)
