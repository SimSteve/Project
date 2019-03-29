import numpy as np
import json
import tensorflow as tf

# seed = 42 TODO seed was 42 now 79
seed = 79
'''
# this function is used for training
def encrypt(inputs):
    dims = np.array(inputs).shape

    permutated_flattened = np.random.RandomState(seed=seed).permutation(inputs.flatten())
    enc_inputs = np.reshape(permutated_flattened, dims)

    return enc_inputs / 255.0
'''

# this function is used when running CW_attack
def encrypt(inputs):
    tf.NoGradient("RandomShuffle")

    shape = tf.shape(inputs)

    permutated_flattened = tf.random.shuffle(tf.reshape(inputs, [-1]), seed=seed)
    enc_inputs = tf.reshape(permutated_flattened, shape)

    return enc_inputs



def print_encryption_details(out):
    out.write("key: {}\n".format(seed))
