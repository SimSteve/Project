import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



inputs = np.array(list(range(784)))
inputs = np.reshape(inputs, (7,7,16))
print(len(inputs.flatten()))

exit()



with tf.Session() as sess:
    size = 1000

    inputs = np.array(list(range(size)))

    seed = 42
    permutation = list(range(size))
    permutation = np.random.RandomState(seed=seed).permutation(permutation)
    reverse = tfp.bijectors.Permute(permutation=permutation)

    enc_inputs = reverse.forward(inputs.flatten())
    dec_inputs = reverse.inverse(enc_inputs)

    np_encrypt = np.random.RandomState(seed=seed).permutation(inputs.flatten())

    print((inputs == sess.run(dec_inputs)).all())
    print((np_encrypt == sess.run(enc_inputs)).all())


