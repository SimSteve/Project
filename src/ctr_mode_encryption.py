import json
import os

import numpy as np
from Crypto.Util import Counter

key = bytes(os.urandom(16))
IV = Counter.new(16 * 8)

mode_of_operation = "AES_CTR_V2"


def prepare_data(x_train, x_test):

    return encrypt_v2(x_train,x_test)


def encrypt_v1(x_train, x_test):
    dims = np.array(x_train).shape

    from Crypto.Cipher import AES
    aes_cipher = AES.new(key, AES.MODE_CTR, counter=IV)

    for image in x_train:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    for image in x_test:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    return x_train, x_test


def encrypt_v2(x_train, x_test):
    dims = np.array(x_train).shape

    from Crypto.Cipher import AES
    aes_cipher = AES.new(key, AES.MODE_CTR, counter=IV)

    for image in x_train:
        np_image = np.array(image)
        blocked_image = blockshaped(np_image, 4, 4)
        flattened = bytes(map(int,[item for block in blocked_image for item in block.flatten().tolist()]))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    for image in x_test:
        np_image = np.array(image)
        blocked_image = blockshaped(np_image, 4, 4)
        flattened = bytes(map(int,[item for block in blocked_image for item in block.flatten().tolist()]))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    return x_train, x_test


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def print_results(test_acc, out, dataset, model, epochs, epoch_accuracies):
    out.write(
        "{} {} {} key is : {} , IV of counter is : {}\n".format(dataset + " " + mode_of_operation + "_model", model,
                                                                test_acc, key, IV))
    out.write(
        "\terror rate : {}%\n".format((1.0 - test_acc) * 100)
    )

    results = {
        "name": dataset + "_" + model + "_" + mode_of_operation,
        "acc": epoch_accuracies,
        "epochs": epochs
    }

    # writing the training results
    with open("../{}_{}_models.json".format(mode_of_operation, dataset), 'a') as j:
        json.dump(results, j)
        j.write('\n')
