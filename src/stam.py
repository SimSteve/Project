import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


def plot_image(image):
    # axes[i].imshow(images[i], cmap=plt.cm.binary)  # row=0, col=0
    plt.imshow(np.reshape(image, (28, 28)))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.show()


img = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])

seed = 42

print(img)
print()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

num_of_examples = 2

x_train = x_train[:num_of_examples]
y_train = y_train[:num_of_examples]

z = [tf.reshape(tf.random.shuffle(tf.reshape(image, [-1]), seed=seed), tf.shape(image)) for image in x_train]

with tf.Session() as sess:
    a = sess.run(z)
    a = [img / 255.0 - 0.5 for img in a]



print(numpy_permutate(image=img, seed=42))

print(tensorflow_permutate(image=img, seed=42))

print(numpy_permutate(numpy_permutate(image=img, seed=42), seed=0))
