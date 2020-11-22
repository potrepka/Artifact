# Artifact

Artifact is a deep learning project that uses convolutional neural networks in
tandem with classical image processing techniques to create unique styles.

## Overview

We begin by using classical image processing techniques to generate modified
images from our original image set. The effects chosen are:
  - Blur
  - Kuwahara
  - Posterize
  - Monochrome (dither from grayscale)
The model, as it is, was not found to be very effective at colorizing images.
Therefore, we use grayscale images as the original image set when training on
the monochrome effect.

Next, we train a CycleGAN model using our modified and original image sets.
Hyperparameters to note, which may differ from the original CycleGAN paper, are
as follows:
  - The generator encoder and decoder use 32, 64, 128 filters instead of 64,
  128, 256. This change was made to satisfy GPU space constraints.
  - The discriminator loss is not scaled by 0.5. Better results were found when
  the discriminator loss was left unscaled.
  - The number of residual layers is 9, as is recommended by the original paper
  for images of size 256 by 256.
  - The number of training iterations is 160,000. The number of validation
  iterations is 40,000. Training is split into 8 epochs.

Finally, once training is complete, we feed a modified image into our model to
obtain a new (fake original) image. Feel free to make observations of artifacts
in your output image and tune the model as you like. Common artifacts are
tiling, light and dark patches, and color distortion. Please note:
  - Using random pairs of images rather than associated pairs yielded better
  results. See `--shuffle` below.
  - Validation loss is not a clear indicator of the degree to which artifacts
  are present.
  - Increasing the number of residual layers to 12 seemed to worsen artifacts.
  - Scaling the discriminator loss by 0.5 seemed to remove some of the color
  distortion at the cost of extra tiling and extra color distortion at the edge
  of light and dark patches.
  - High-contrast, noisy images yield the best results. Likewise, images with
  flat coloring yield the worst results.

## Setup

First, install the required dependencies:
```
pip install -r requirements.txt --upgrade
```

Once installation is complete, you can download and unzip the
[COCO](http://images.cocodataset.org/zips/unlabeled2017.zip) dataset:
```
python download.py
python unzip.py
```

## Data Processing

Download [ImageMagick](https://imagemagick.org/). Please note that the commands
below will take a very long time to complete for the full dataset.

Before processing, it is highly recommended to first resize and crop the entire
set of images:
```
magick mogrify -resize 256x256^ -gravity center -extent 256x256 "data/COCO/*.jpg"
```

Next, copy the images into new folders:
```
cp -r data/COCO data/COCO_blur_2
cp -r data/COCO data/COCO_kuwahara_2
cp -r data/COCO data/COCO_posterize_8

cp -r data/COCO data/COCO_gray
cp -r data/COCO data/COCO_monochrome
```

Lastly, apply effects to the copied images:
```
magick mogrify -blur 0x2 "data/COCO_blur_2/*.jpg"
magick mogrify -kuwahara 2 "data/COCO_kuwahara_2/*.jpg"
magick mogrify -posterize 8 "data/COCO_posterize_8/*.jpg"

magick mogrify -colorspace gray "data/COCO_gray/*.jpg"
magick mogrify -monochrome "data/COCO_monochrome/*.jpg"
```

## Training

To train your model, run one of the following commands:
```
python train.py --modified data/COCO_blur_2 --prefix blur_2 --shuffle
python train.py --modified data/COCO_kuwahara_2 --prefix kuwahara_2 --shuffle
python train.py --modified data/COCO_posterize_8 --prefix posterize_8 --shuffle

python train.py --modified data/COCO_monochrome --original data/COCO_gray --prefix monochrome --shuffle
```

Please consult the help documentation for more information:
```
python train.py --help
```

## Evaluation

Once training is complete, you can use a checkpoint and a processed image to
generate a new (fake original) image:
```
python evaluate.py --checkpoint checkpoints/blur_2_epoch=007.ckpt --input /PATH/TO/INPUT/IMAGE --fake
python evaluate.py --checkpoint checkpoints/kuwahara_2_epoch=007.ckpt --input /PATH/TO/INPUT/IMAGE --fake
python evaluate.py --checkpoint checkpoints/posterize_8_epoch=007.ckpt --input /PATH/TO/INPUT/IMAGE --fake

python evaluate.py --checkpoint checkpoints/monochrome_epoch=007.ckpt --input /PATH/TO/INPUT/IMAGE --fake
```

Additionally, you can view the parameters of your model like so:
```
python evaluate.py --checkpoint /PATH/TO/MODEL/CHECKPOINT
```

Please consult the help documentation for more information:
```
python evaluate.py --help
```

## Logging

You can view log data from your training session using TensorBoard:
```
tensorboard --logdir lightning_logs/version_0
```

## Status

Run the following script for information about PyTorch and your graphics card:
```
python info.py
```

## License

MIT License

## Bibliography

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```
