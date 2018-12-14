import tensorflow as tf
from pathlib import Path


class Model:
    def build(self, in_shape):
        pass

    def train(self, x_train, y_train, ep):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=ep)

    def compile(self):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return  self.model.predict(x_test)

    def save(self, m_file):
        tf.keras.models.save_model(self.model, '../saved_models/' + m_file)

    def load(self, m_file):
        if Path('../saved_models/' + m_file).is_file():
            self.model = tf.keras.models.load_model('../saved_models/' + m_file)
            self.compile()
        else:
            print("WARNING! No saved model found, train one from scratch.")
            self.model = None
        return self

    def summary(self):
        self.model.summary()


class FGSM(Model):
    def build(self, in_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=8, input_shape=in_shape, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=128, kernel_size=6, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])


class CW_1(Model):
    def build(self, in_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=in_shape, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])


class CW_2(Model):
    def build(self, in_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=in_shape, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
