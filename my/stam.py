import tensorflow as tf
import numpy as np
import my.src.my_ecb as e
import my.src.my_Models as m
import my.src.my_unencrypted as u
import tensorflow.keras.backend as K


model = m.Encrypted_Model(encrypt=e.encrypt_v2)
#model = m.Encrypted_Model(encrypt=u.encrypt)
model.load("fashion_mnist_CW_2_ECB_V2_weights")
#model.load("fashion_mnist_CW_2_UNENCRYPTED")

#exit()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#x_test = x_test / 255.0
x_test = x_test / 1.0
dims = np.array(x_test).shape
if len(dims) != 4:
    # expanding the images to get a third dimension (needed for conv layers)
    x_test = np.expand_dims(x_test, -1)


inp = model.model.input
outputs = model.model.layers[-2].output
print(outputs)
func = K.function([inp], [outputs])
layer_outs = func([x_test[:1]])
print(layer_outs)

inp = model.model.input
outputs = model.model.layers[-1].output
print(outputs)
func = K.function([inp], [outputs])
layer_outs = func([x_test[:1]])
print(layer_outs)

exit()

test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_acc)