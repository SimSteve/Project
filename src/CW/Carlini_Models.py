import tensorflow as tf
from tensorflow.keras.models import Sequential
from pathlib import Path


class Encrypted_Model():
    def __init__(self, encrypt=lambda a: a):
        self.encrypt = encrypt

    def train(self, x_train, y_train, ep):
        for i, image in enumerate(x_train):
            x_train[i] = self.encrypt(image)

        h = self.model.fit(x_train, y_train, epochs=ep)
        return h.history['loss'], h.history['acc'], list(range(ep))

    def evaluate(self, x_test, y_test):
        for i, image in enumerate(x_test):
            x_test[i] = self.encrypt(image)

        return self.model.evaluate(x_test, y_test)

    def predict(self, image):
        enc_img = self.encrypt(image)
        return self.model(enc_img)

    def save(self, m_file):
        tf.keras.models.Model.save_weights(filepath='saved_weights/{}.h5'.format(m_file), save_format='h5')

    def load(self, m_file):
        if Path('saved_weights/{}.h5'.format(m_file)).is_file():
            self.model.load_weights(filepath='saved_weights/{}.h5'.format(m_file))
        else:
            print("WARNING! No saved model found, train one from scratch.")
            self.model = None
        return self

    def summary(self):
        self.model.summary()


class CW_1(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a: a):
        super(CW_1, self).__init__(encrypt)

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        self.model = Sequential()

        self.model.add(
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, name="before-softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


class CW_2(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a: a):
        super(CW_2, self).__init__(encrypt)

        self.model = Sequential()

        self.model.add(
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, name="before-softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
