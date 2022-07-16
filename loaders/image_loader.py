import glob
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


def image_loader(path):
    training_images = glob.glob(path)

    output_images = []

    for filename in sorted(training_images):
        image = np.asarray(Image.open(filename))
        output_images.append(image)

    return output_images