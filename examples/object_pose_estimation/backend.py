import numpy as np
from paz.backend.image import gaussian_image_blur, load_image, show_image


def apply_gaussian_noise(image, mean=0, variance=0.1):
    standard_deviation = np.sqrt(variance)
    gaussian_noise = np.random.normal(mean, standard_deviation, image.shape)
    image = image + gaussian_noise
    image = image.astype('uint8')
    return image


def apply_speckle_noise(image, mean=0, variance=0.1):
    standard_deviation = np.sqrt(variance)
    gaussian_noise = np.random.normal(mean, standard_deviation, image.shape)
    image = image + (image * gaussian_noise)
    image = image.astype('uint8')
    return image


def apply_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    image = np.random.poisson(image * vals) / float(vals)
    return image.astype('uint8')


def add_salt_and_pepper(image):
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out.astype('uint8')


"""
image = load_image('solar_panel.jpg')
show_image(image)
show_image(apply_gaussian_noise(image))
show_image(apply_speckle_noise(image, 0, 0.05))
show_image(apply_poisson_noise(image))
show_image(add_salt_and_pepper(image))
"""
