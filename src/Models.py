import tensorflow as tf
from tensorflow.keras.models import Sequential
from pathlib import Path


class Encrypted_Model():
    def __init__(self, encrypt=lambda a:a):
        self.encrypt = encrypt

    def train(self, x, y_train, ep):
        x_train = x.copy()
        for i,image in enumerate(x_train):
            x_train[i] = self.encrypt(image)

        h = self.model.fit(x_train, y_train, epochs=ep)
        return h.history['loss'], h.history['acc'], list(range(ep))

    def evaluate(self, x, y_test):
        x_test = x.copy()
        for i,image in enumerate(x_test):
            x_test[i] = self.encrypt(image)

        return self.model.evaluate(x_test, y_test)

    def predict(self, image):
        enc_img = self.encrypt(image)
        return self.model.predict(enc_img)

    def save(self, m_file):
        self.model.save_weights(filepath='saved_weights/{}.h5'.format(m_file))

    def load(self, m_file):
        if Path('saved_weights/{}.h5'.format(m_file)).is_file():
            self.model.load_weights(filepath='saved_weights/{}.h5'.format(m_file))
        else:
            print("WARNING! No saved model found, train one from scratch.")
            self.model = None
        return self

    def summary(self):
        self.model.summary()


class FGSM(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(FGSM, self).__init__(encrypt)

        self.model = Sequential()
        
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=8, input_shape=input_shape, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=6, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, name="before-softmax"))
        self.model.add(tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


class CW_1(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(CW_1, self).__init__(encrypt)

        self.model = Sequential()

        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu))
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
        self.model.add(tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


class CW_2(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(CW_2, self).__init__(encrypt)

        self.model = Sequential()
        
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu))
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
        self.model.add(tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax"))

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
