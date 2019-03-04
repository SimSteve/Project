import numpy as np
import json

mode = "MY_PERMUTATED"

seed = 42


def encrypt(inputs):
    dims = np.array(inputs).shape

    permutated_flattened = np.random.RandomState(seed=seed).permutation(inputs.flatten())
    enc_inputs = np.reshape(permutated_flattened, dims)

    return enc_inputs


def print_encryption_details(out):
    out.write("key: {}".format(seed))