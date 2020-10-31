from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def process_image(image, size=256, crop_size=224):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    :param image:
    :param size:
    :param crop_size:
    :return: numpy array
    """
    # Ref: for image resizing with respect to ration https://gist.github.com/tomvon/ae288482869b495201a0
    image = Image.open(image).convert("RGB")

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    og_size = image.size

    width, height = image.size

    # Resize our image while keeping our aspect ration
    width_percent = (size / float(width))
    height = int((float(height) * float(width_percent)))
    image = image.resize((size, height))
    print(f'Image resized to: {image.size}, from: {og_size}')

    # crop our image from the middle out
    width, height = image.size
    left = (width - crop_size) / 2
    upper = (height - crop_size) / 2
    right = left + crop_size
    lower = upper + crop_size
    print(f'left: {left}, upper: {upper}, right: {right}, lower: {lower}')
    image = image.crop((left, upper, right, lower))

    # convert to float array in numpy
    np_image = np.array(image) / 255

    # subtract means from each color channel and divide by std deviation
    np_image = (np_image - mean) / std

    # finally, transpose the dimensions.  PyTorch expects the oclor channel to be the first dimension
    # buts its the third in the PIL image and numpy array.
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax