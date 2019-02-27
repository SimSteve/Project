import json
import os
from Crypto.Cipher import AES
import numpy as np

key = bytes(os.urandom(16))

mode_of_operation = "MY_AES_ECB_V2"


def encrypt(inputs):

    return encrypt_v2(inputs)


def encrypt_v1(inputs):
    dims = np.array(inputs).shape

    aes_cipher = AES.new(key, AES.MODE_ECB)

    flattened = bytes(map(int, inputs.flatten().tolist()))
    aes_flattened = list(aes_cipher.encrypt(flattened))
    enc_inputs = np.reshape(aes_flattened, dims)

    return enc_inputs


def encrypt_v2(inputs):
    dims = np.array(inputs).shape

    aes_cipher = AES.new(key, AES.MODE_ECB)

    np_image = np.array(image)
    blocked_image = blockshaped(np_image, 4, 4)
    flattened = bytes(map(int,[item for block in blocked_image for item in block.flatten().tolist()]))
    aes_flattened = list(aes_cipher.encrypt(flattened))
    enc_inputs = np.reshape(aes_flattened, dims)

    return enc_inputs


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
        "{} {} {} key is : {}\n".format(dataset + " " + mode_of_operation + "_model", model,
                                                                test_acc, key))
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