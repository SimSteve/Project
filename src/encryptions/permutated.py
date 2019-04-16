import numpy as np
import json
import tensorflow as tf
import tensorflow_probability as tfp

seed = 42
permutation_indexes = np.random.RandomState(seed=seed).permutation(list(range(784)))
reverse = tfp.bijectors.Permute(permutation=permutation_indexes)

# # this function is used for training
# def encrypt(inputs):
#     dims = np.array(inputs).shape
#
#     permutated_flattened = np.random.RandomState(seed=seed).permutation(inputs.flatten())
#     enc_inputs = np.reshape(permutated_flattened, dims)
#
#     return enc_inputs / 255.0 - 0.5


# def encrypt(inputs):
#     dims = np.array(inputs).shape
#
#     permutated_flattened = np.random.RandomState(seed=seed).permutation(inputs.flatten())
#     enc_inputs = np.reshape(permutated_flattened, dims)
#
#     return enc_inputs

def encrypt(inputs):
    shape = tf.shape(inputs)

    enc_inputs = reverse.forward(tf.reshape(inputs, [-1]))
    dec_inputs = reverse.inverse(enc_inputs)

    enc_inputs = tf.reshape(enc_inputs, shape)
    # dec_inputs = tf.reshape(dec_inputs, shape)

    return enc_inputs

#
# # this function is used when running CW_attack
# def encrypt(inputs):
#     tf.NoGradient("RandomShuffle")
#
#     shape = tf.shape(inputs)
#
#     permutated_flattened = tf.random.shuffle(tf.reshape(inputs, [-1]), seed=seed)
#     enc_inputs = tf.reshape(permutated_flattened, shape)
#
#     return enc_inputs

def print_encryption_details(out):
    out.write("key: {}\n".format(seed))
