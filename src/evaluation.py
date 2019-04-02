import src.Models as m
import tensorflow as tf
import src.encryptions.permutated as e
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 1.0, x_test / 1.0

dims = np.array(x_train).shape

if len(dims) != 4:
    # expanding the images to get a third dimension (needed for conv layers)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

input_shape = np.array(x_train[0]).shape

model = m.CW_1(input_shape, e.encrypt)

model.load("mnist_CW_1_PERMUTATED")

test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_acc * 100)
'''
good = 0.0
bad = 0.0
for _ in range(2):
    for i in range(len(y_test)):
        pred = np.argmax(model.predict(np.reshape(x_test[i], (1,28,28,1))))
        real = y_test[i]
        good += pred == real
        bad += pred != real

print((good / (good + bad))* 100)
'''