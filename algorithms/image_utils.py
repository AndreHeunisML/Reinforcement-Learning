
from scipy.misc import imresize
import numpy as np


def downsample_image(image, target_image_size):
    """

    :param image:
    :param target_image_size:
    :return:
    """
    B = image[31:195]  # select the important parts of the image
    B = B.mean(axis=2)  # convert to grayscale

    # downsample image
    # changing aspect ratio doesn't significantly distort the image
    # nearest neighbor interpolation produces a much sharper image
    # than default bilinear
    B = imresize(B, size=(target_image_size, target_image_size), interp='nearest')
    return B


def update_state(state, obs):
    obs_small = downsample_image(obs)
    return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)