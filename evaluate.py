from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from image.image import ImageReader
from image.image import ImageWriter
from models.cyclegan.model import CycleGAN
import os
import torch

def load_model_and_freeze(checkpoint_path):
    map_location = lambda storage, loc: storage
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    try:
        hyper_parameters = checkpoint['hyper_parameters']
        model = CycleGAN(None, None, **hyper_parameters)
    except KeyError:
        print('Warning: No hyperparameters found. Using defaults.')
        model = CycleGAN(None, None)
    model.load_state_dict(checkpoint['state_dict'])
    model.on_load_checkpoint(checkpoint)
    model.eval()
    model.freeze()
    return model

def main(params):
    model = load_model_and_freeze(params.checkpoint)

    if params.input is None:
        print(model.hparams)
    else:
        dir = os.path.dirname(params.input)
        base = os.path.splitext(os.path.basename(params.input))[0]
        suffix = os.path.splitext(os.path.basename(params.checkpoint))[0]
        extension = params.extension

        read_image = ImageReader(channels=model.in_channels, resize=False)
        write_image = ImageWriter(channels=model.out_channels, resize=False)

        image = read_image(params.input)
        image = torch.unsqueeze(image, 0)

        if params.fake or params.cycle or params.diff or params.all:
            fake = model.gen_a_to_b(image)

        if params.cycle or params.diff or params.all:
            cycle = model.gen_b_to_a(fake)

        if params.diff or params.all:
            diff = cycle - image

        if params.original or params.all:
            original_name = base + '_original_' + suffix
            original_path = os.path.join(dir, original_name + '.' + extension)
            write_image(original_path, torch.squeeze(image, 0))

        if params.fake or params.all:
            fake_name = base + '_fake_' + suffix
            fake_path = os.path.join(dir, fake_name + '.' + extension)
            write_image(fake_path, torch.squeeze(fake, 0))

        if params.cycle or params.all:
            cycle_name = base + '_cycle_' + suffix
            cycle_path = os.path.join(dir, cycle_name + '.' + extension)
            write_image(cycle_path, torch.squeeze(cycle, 0))

        if params.diff or params.all:
            diff_name = base + '_diff_' + suffix
            diff_path = os.path.join(dir, diff_name + '.' + extension)
            write_image(diff_path, torch.squeeze(diff, 0))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--checkpoint', help='path to model checkpoint', required=True)
    required.add_argument('--input', help='path to modified image')
    optional.add_argument('--extension', default='png', help='extension to use')
    optional.add_argument('--all', dest='all', action='store_true', help='write all images')
    optional.add_argument('--original', dest='original', action='store_true', help='write original image')
    optional.add_argument('--fake', dest='fake', action='store_true', help='write fake image')
    optional.add_argument('--cycle', dest='cycle', action='store_true', help='write cycle image')
    optional.add_argument('--diff', dest='diff', action='store_true', help='write difference between cycle and original as image')
    optional.set_defaults(all=False)
    optional.set_defaults(original=False)
    optional.set_defaults(fake=False)
    optional.set_defaults(cycle=False)
    optional.set_defaults(diff=False)
    parser._action_groups.append(optional)

    params = parser.parse_args()
    main(params)
