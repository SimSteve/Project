import my.src.my_Models as m
import my.src.my_unencrypted as e

name = "mnist_FGSM_UNENCRYPTED"

model = m.Encrypted_Model(encrypt=e.encrypt)
model.load(name)

model.model.save_weights('saved_weights/{}.h5'.format(name))
