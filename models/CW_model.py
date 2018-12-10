import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
class_names = []

class_types = {'fashion_mnist':fashion_mnist_classes, 'mnist':mnist_classes}
data_types = {'fashion_mnist':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist}
file_types = {'fashion_mnist':'fashion_mnist_CW_model', 'mnist':'mnist_CW_model'}


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]), color=color)
    plt.show()


class FGSM_model():
    def __init__(self,m_file):
        self.model_file = m_file

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1), activation=tf.nn.relu),
            # tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(28, 28, 1), activation=tf.nn.relu),

            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu),
            # tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),

            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),
            # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),
            # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu),

            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            # tf.keras.layers.Dense(256, activation=tf.nn.relu),

            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            # tf.keras.layers.Dense(256, activation=tf.nn.relu),

            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    def train(self, x_train, y_train, ep):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=ep)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return  self.model.predict(x_test)

    def save(self):
        tf.keras.models.save_model(self.model, '../saved_models/' + self.model_file)

    def load(self):
        if Path('../saved_models/' + self.model_file).is_file():
            self.model = tf.keras.models.load_model('../saved_models/' + self.model_file)
        else:
            self.model = None
        return self

    def summary(self):
        self.model.summary()


def create_model(m_file, data):
    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = FGSM_model(m_file)

    # building the networks' structure
    model.build()

    # training
    model.train(x_train, y_train, ep=5)

    # evaluating
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    # saving model
    model.save()


def test(m_file, data):
    # loading a saved model
    model = FGSM_model(m_file).load()

    _, (x_test, y_test) = data.load_data()
    x_test = x_test / 255.0

    x_test = np.expand_dims(x_test, -1)
    predictions = model.predict(x_test)
    x_test = x_test[:, :, :, 0]

    plt.figure(figsize=(6, 3))
    plot_image(i=969, predictions_array=predictions, true_label=y_test, img=x_test)


if __name__ == '__main__':
    type = 'mnist'

    m_file = file_types[type]
    data = data_types[type]
    class_names = class_types[type]

    create_model(m_file, data)
    test(m_file, data)
