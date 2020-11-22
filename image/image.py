import cv2
import numpy as np
from skimage import transform, util
import torch

class ImageTransform:
    def __init__(self, size=256, channels=3, resize=True):
        self.size = size
        self.channels = channels
        self.resize = resize

    def __call__(self, image, reverse=False):
        if reverse:
            # Scale
            image = torch.clamp((image + 1.0) * 128, 0.0, 255.0)
            # Transpose dimensions
            image = image.T
            # Use a standard number of channels: 1, 3, or 4
            if image.shape[2] == 0 or image.shape[2] == 2:
                # Pad channels
                image = util.pad(
                    image,
                    ((0, 1), (0, 0), (0, 0)),
                    mode='constant'
                )
            elif image.shape[2] > 4:
                # Crop channels
                image = image[:, :, :4]
            # Convert to numpy array
            image = image.numpy()
        else:
            if self.resize:
                if image.shape[0] != self.size or image.shape[1] != self.size:
                    # Calculate padding
                    tb_pad = 0
                    lr_pad = 0
                    if image.shape[1] > image.shape[0]:
                        # Landscape
                        tb_pad = (image.shape[1] - image.shape[0]) // 2
                    else:
                        # Portrait
                        lr_pad = (image.shape[0] - image.shape[1]) // 2
                    # Pad height and width
                    image = util.pad(
                        image,
                        ((tb_pad, tb_pad), (lr_pad, lr_pad), (0, 0)),
                        mode='reflect'
                    )
                if image.shape[0] != self.size or image.shape[1] != self.size:
                    # Resize height and width
                    image = transform.resize(image, (self.size, self.size))
            if image.shape[2] < self.channels:
                # Pad channels
                c_pad = self.channels - image.shape[2]
                image = util.pad(
                    image,
                    ((0, 0), (0, 0), (0, c_pad)),
                    mode='wrap'
                )
            elif image.shape[2] > self.channels:
                # Crop channels
                image = image[:, :, :self.channels]
            # Transpose dimensions
            image = image.T
            # Scale
            image = image * 0.0078125 - 1.0
            # Convert to float tensor
            image = torch.from_numpy(image).float()
        return image

class Image:
    def __init__(self, size=256, channels=3, resize=True):
        self.transform = ImageTransform(size, channels, resize)

class ImageReader(Image):
    def __init__(self, size=256, channels=3, resize=True):
        super(ImageReader, self).__init__(size, channels, resize)

    def __call__(self, path):
        image = cv2.imread(path)
        image = self.transform(image)
        return image

class ImageWriter(Image):
    def __init__(self, size=256, channels=3, resize=True):
        super(ImageWriter, self).__init__(size, channels, resize)

    def __call__(self, path, image):
        image = self.transform(image, reverse=True)
        image = image.astype(np.uint8)
        cv2.imwrite(path, image)
