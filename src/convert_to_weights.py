import src.Models_old as m
import src.encryptions.unencrypted as e

name = "mnist_CW_2_PERMUTATED"

model = m.Encrypted_Model(encrypt=e.encrypt)
model.load(name)

model.model.save_weights('saved_weights/{}.h5'.format(name))
