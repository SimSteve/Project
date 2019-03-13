import json
import os
from Crypto.Cipher import AES
import numpy as np

#key = bytes(os.urandom(16))
key = b'&\xcc\xa4\xaa\x88\xbc\xad\xcf\x0f\xe9`\xe1w]\x1eo'


def encrypt(inputs):

    return encrypt_v2(inputs)


def encrypt_v1(inputs):
    dims = np.array(inputs).shape

    aes_cipher = AES.new(key, AES.MODE_ECB)

    flattened = bytes(map(int, inputs.flatten().tolist()))
    aes_flattened = list(aes_cipher.encrypt(flattened))
    enc_inputs = np.reshape(aes_flattened, dims)

    return enc_inputs / 255.0


def encrypt_v2(inputs):
    dims = np.array(inputs).shape

    aes_cipher = AES.new(key, AES.MODE_ECB)

    np_image = np.array(inputs)
    blocked_image = blockshaped(np_image, 4, 4)
    flattened = bytes(map(int,[item for block in blocked_image for item in block.flatten().tolist()]))
    aes_flattened = list(aes_cipher.encrypt(flattened))
    enc_inputs = np.reshape(aes_flattened, dims)
    return enc_inputs / 255.0


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w, d = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def print_encryption_details(out):
    out.write("key: {}\n".format(key))
