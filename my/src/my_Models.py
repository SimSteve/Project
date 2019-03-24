import tensorflow as tf
from pathlib import Path


class Encrypted_Model():
    def __init__(self, encrypt=lambda a:a):
        self.encrypt = encrypt

    def train(self, x_train, y_train, ep):
        for i,image in enumerate(x_train):
            x_train[i] = self.encrypt(image)

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        h = self.model.fit(x_train, y_train, epochs=ep)
        return h.history['loss'], h.history['acc'], list(range(ep))

    def compile(self):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def evaluate(self, x_test, y_test):
        #for i,image in enumerate(x_test):
        #    x_test[i] = self.encrypt(image)

        return self.model.evaluate(x_test, y_test)

    def predict(self, image):
        enc_img = self.encrypt(image)
        return self.model.predict(enc_img)

    def save(self, m_file):
        tf.keras.models.save_model(self.model, 'saved_models/' + m_file)

    def load(self, m_file):
        if Path('saved_models/' + m_file).is_file():
            self.model = tf.keras.models.load_model('saved_models/' + m_file)
            self.compile()
        else:
            print("WARNING! No saved model found, train one from scratch.")
            exit()
        return self

    def summary(self):
        self.model.summary()


class FGSM(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(FGSM, self).__init__(encrypt)
        self.inputs = tf.keras.layers.Input(input_shape)
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=8, input_shape=input_shape, activation=tf.nn.relu)(self.inputs)
        self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=6, activation=tf.nn.relu)(self.conv_1)
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation=tf.nn.relu)(self.conv_2)
        self.flatten = tf.keras.layers.Flatten()(self.conv_3)
        self.dense = tf.keras.layers.Dense(10, name="before-softmax")(self.flatten)
        self.output = tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax")(self.dense)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.output)


class CW_1(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(CW_1, self).__init__(encrypt)
        self.inputs = tf.keras.layers.Input(input_shape)
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu)(self.inputs)
        self.conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)(self.conv_1)
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=2)(self.conv_2)
        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(self.pool_1)
        self.conv_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(self.conv_3)
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=2)(self.conv_4)
        self.flatten = tf.keras.layers.Flatten()(self.pool_2)
        self.dense_1 = tf.keras.layers.Dense(200, activation=tf.nn.relu)(self.flatten)
        self.dropout = tf.keras.layers.Dropout(0.2)(self.dense_1)
        self.dense_2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)(self.dropout)
        self.dense_3 = tf.keras.layers.Dense(10, name="before-softmax")(self.dense_2)
        self.output = tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax")(self.dense_3)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.output)


class CW_2(Encrypted_Model):
    def __init__(self, input_shape, encrypt=lambda a:a):
        super(CW_2, self).__init__(encrypt)
        self.inputs = tf.keras.layers.Input(input_shape)
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=input_shape, activation=tf.nn.relu)(
            self.inputs)
        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu)(self.conv_1)
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=2)(self.conv_2)
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(self.pool_1)
        self.conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu)(self.conv_3)
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=2)(self.conv_4)
        self.flatten = tf.keras.layers.Flatten()(self.pool_2)
        self.dense_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(self.flatten)
        self.dropout = tf.keras.layers.Dropout(0.2)(self.dense_1)
        self.dense_2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(self.dropout)
        self.dense_3 = tf.keras.layers.Dense(10, name="before-softmax")(self.dense_2)
        self.output = tf.keras.layers.Activation(activation=tf.nn.softmax, name="softmax")(self.dense_3)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.output)
