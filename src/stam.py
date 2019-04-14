import tensorflow as tf
import numpy as np


def numpy_permutate(image, seed):
    dims = np.array(image).shape

    permutated_flattened = np.random.RandomState(seed=seed).permutation(image.flatten())
    enc_inputs = np.reshape(permutated_flattened, dims)

    return enc_inputs


def tensorflow_permutate(image, seed):
    with tf.Session() as sess:
        shape = tf.shape(image)

        permutated_flattened = tf.random.shuffle(tf.reshape(image, [-1]), seed=seed)
        enc_inputs = tf.reshape(permutated_flattened, shape)

        return sess.run(enc_inputs)


img = np.array([[1.0,2,3],[4,5,6],[7,8,9]])

print(img)
print()

print(numpy_permutate(image=img, seed=42))

print(tensorflow_permutate(image=img, seed=42))

print(numpy_permutate(numpy_permutate(image=img, seed=42), seed=0))

