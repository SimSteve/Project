import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_image(img):
    '''
    Draws the image. ONLY for greyscale images (the third dimension should be 1)
    :param predictions: vector of probablities
    :param true_label: the true label
    :param img: the image itself
    :return:
    '''
    # in order to plot the image, we need a 2D array
    if len(np.array(img).shape) == 3:
        img = img[:, :, 0]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    plt.show()

if __name__ == '__main__':
    img = np.loadtxt(sys.argv[1])
    plot_image(img)
