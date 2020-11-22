from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import common.data as data
import common.setup as setup
import datetime
from image.coco import COCO
from models.cyclegan.model import CycleGAN
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.parsing import AttributeDict

def main(params):
    # Display date and time
    print(datetime.datetime.now())
    print(AttributeDict(vars(params)))

    # Load training data
    train_dataset = COCO(
        directory_a=params.modified,
        directory_b=params.original,
        num_training=params.num_training,
        num_validation=params.num_validation,
        size=params.size,
        channels=params.channels,
        shuffle=params.shuffle,
        cache=params.cache,
        validation=False
    )
    train_loader = setup.load(train_dataset, params.batch_size, False)

    # Load validation data
    val_dataset = COCO(
        directory_a=params.modified,
        directory_b=params.original,
        num_training=params.num_training,
        num_validation=params.num_validation,
        size=params.size,
        channels=params.channels,
        shuffle=params.shuffle,
        cache=params.cache,
        validation=True
    )
    val_loader = setup.load(val_dataset, params.batch_size, False)

    # Create model
    model = CycleGAN(
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=params.batch_size,
        iterations=params.epochs * params.num_training,
        in_channels=params.channels,
        out_channels=params.channels,
        g_filters=params.g_filters,
        d_filters=params.d_filters,
        residual_blocks=params.residual_blocks,
        dropout=params.dropout,
        skip=params.skip,
        learning_rate=params.lr,
        beta_1=params.b1,
        beta_2=params.b2,
        init_type=params.init_type,
        init_scale=params.init_scale,
        pool_size_a=params.pool_size_a,
        pool_size_b=params.pool_size_b,
        lambda_dis_a=params.lambda_dis_a,
        lambda_dis_b=params.lambda_dis_b,
        lambda_gen_a=params.lambda_gen_a,
        lambda_gen_b=params.lambda_gen_b,
        lambda_cycle_a=params.lambda_cycle_a,
        lambda_cycle_b=params.lambda_cycle_b,
        lambda_id_a=params.lambda_id_a,
        lambda_id_b=params.lambda_id_b,
        shuffle=params.shuffle
    )

    # Set up trainer
    checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename=params.prefix + '_{epoch:03d}',
        save_top_k=1 if params.save_top_only else -1,
        monitor='val_loss',
        verbose=True
    )
    if setup.cuda_is_available():
        trainer = Trainer(
            accelerator='ddp',
            gpus=setup.cuda_device_count(),
            callbacks=[checkpoint],
            max_epochs=params.epochs,
            precision=params.precision
        )
    else:
        trainer = Trainer(
            callbacks=[checkpoint],
            max_epochs=params.epochs,
            precision=params.precision
        )

    # Train
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prefix', default='', help='prefix for checkpoint file name')
    parser.add_argument('--modified', default=data.directory, help='directory of modified images')
    parser.add_argument('--original', default=data.directory, help='directory of original images')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--precision', type=int, default=16, help='floating point precision')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train')
    parser.add_argument('--num_training', type=int, default=20000, help='number of training examples per epoch')
    parser.add_argument('--num_validation', type=int, default=5000, help='number of validation examples per epoch')
    parser.add_argument('--save_top_only', dest='save_top_only', action='store_true', help='only keep checkpoint with least validation loss')
    parser.add_argument('--size', type=int, default=256, help='image size to use')
    parser.add_argument('--channels', type=int, default=3, help='number of channels to use')
    parser.add_argument('--g_filters', type=int, default=32, help='number of base filters in the generator')
    parser.add_argument('--d_filters', type=int, default=64, help='number of base filters in the discriminator')
    parser.add_argument('--residual_blocks', type=int, default=9, help='number of residual blocks')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--dropout', dest='dropout', action='store_true', help='use dropout in residual blocks')
    parser.add_argument('--skip', dest='skip', action='store_true', help='use skip connections in generator')
    parser.add_argument('--init_type', default='normal', help='weight initialization type: normal, xavier, kaiming, orthogonal')
    parser.add_argument('--init_scale', type=float, default=0.02, help='weight initialization scale')
    parser.add_argument('--pool_size_a', type=int, default=50, help='size of data pool A, which stores fake images for discriminator')
    parser.add_argument('--pool_size_b', type=int, default=50, help='size of data pool B, which stores fake images for discriminator')
    parser.add_argument('--lambda_dis_a', type=float, default=1.0, help='coefficient for discriminator loss A')
    parser.add_argument('--lambda_dis_b', type=float, default=1.0, help='coefficient for discriminator loss B')
    parser.add_argument('--lambda_gen_a', type=float, default=1.0, help='coefficient for generator loss A')
    parser.add_argument('--lambda_gen_b', type=float, default=1.0, help='coefficient for generator loss B')
    parser.add_argument('--lambda_cycle_a', type=float, default=10.0, help='coefficient for cycle loss A')
    parser.add_argument('--lambda_cycle_b', type=float, default=10.0, help='coefficient for cycle loss B')
    parser.add_argument('--lambda_id_a', type=float, default=5.0, help='coefficient for identity loss A')
    parser.add_argument('--lambda_id_b', type=float, default=5.0, help='coefficient for identity loss B')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='decorrelate image pairs')
    parser.add_argument('--cache', dest='cache', action='store_true', help='cache images after loading')
    parser.set_defaults(save_top_only=False)
    parser.set_defaults(dropout=False)
    parser.set_defaults(skip=False)
    parser.set_defaults(shuffle=False)
    parser.set_defaults(cache=False)

    params = parser.parse_args()
    main(params)
