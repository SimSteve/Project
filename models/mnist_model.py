import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


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

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)
    plt.show()


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=8 * 8, activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters=128, kernel_size=6 * 6, activation=tf.nn.relu),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5 * 5, activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = model.predict(x_test)
i = 6666
plt.figure(figsize=(6, 3))
plot_image(i, predictions, y_test, x_test)
