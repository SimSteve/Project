import numpy as np


def pad(img, number_of_paddings, padder=0):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    return np.pad(img, number_of_paddings, pad_with, padder=padder)

# a = np.arange(9)
# a = a.reshape((3, 3))
# print(pad(a,3,7))