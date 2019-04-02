import tensorflow as tf
import numpy as np

seed = 42


def numpy_permutate(image):
    dims = np.array(image).shape

    permutated_flattened = np.random.RandomState(seed=seed).permutation(image.flatten())
    enc_inputs = np.reshape(permutated_flattened, dims)

    return enc_inputs


def tensorflow_permutate(image):
    with tf.Session() as sess:
        shape = tf.shape(image)

        permutated_flattened = tf.random.shuffle(tf.reshape(image, [-1]), seed=seed)
        enc_inputs = tf.reshape(permutated_flattened, shape)

        return sess.run(enc_inputs)


img = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(img)
print()

print(numpy_permutate(image=img))

print(tensorflow_permutate(image=img))





