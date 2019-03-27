import tensorflow as tf
import numpy as np
import src.encryptions.unencrypted as e
import src.Models_old as m
import tensorflow.keras.backend as K


model = m.Encrypted_Model(encrypt=e.encrypt)
#model = m.Encrypted_Model(encrypt=u.encrypt)
model.load("mnist_CW_1_UNENCRYPTED")
#model.load("fashion_mnist_CW_2_UNENCRYPTED")

#exit()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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
layer_outs = func([x_test[9000:9001]])
print(layer_outs)
print(layer_outs[0][0])
exit()
inp = model.model.input
outputs = model.model.layers[-1].output
print(outputs)
func = K.function([inp], [outputs])
layer_outs = func([x_test[:1]])
print(layer_outs)

exit()

test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_acc)