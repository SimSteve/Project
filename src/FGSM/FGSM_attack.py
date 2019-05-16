import tensorflow as tf
import numpy as np
import src.Models as m
from src.FGSM.cleverhans.cleverhans.attacks.fast_gradient_method import FastGradientMethod
from src.FGSM.cleverhans.cleverhans.utils_keras import KerasModelWrapper

FILE = "FILE"


def attack(img, label, model_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_shape = np.array(img[0]).shape
        model = m.modelB(input_shape)
        model.load(model_name)

        wrap = KerasModelWrapper(model)
        fgsm = FastGradientMethod(wrap, sess=sess)
        fgsm_params = {'eps': 0.3,
                       'clip_min': 0.,
                       'clip_max': 1.}

        adv = fgsm.generate_np(img, **fgsm_params)

        return adv
