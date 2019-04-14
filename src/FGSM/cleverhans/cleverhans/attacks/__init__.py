"""
The Attack class, providing a universal abstract interface describing attacks, and many implementations of it.
"""
from six.moves import xrange
import tensorflow as tf

from src.FGSM.cleverhans.cleverhans import utils
from src.FGSM.cleverhans.cleverhans.attacks.attack import Attack
from src.FGSM.cleverhans.cleverhans.attacks.elastic_net_method import ElasticNetMethod
from src.FGSM.cleverhans.cleverhans.attacks.lbfgs import LBFGS
from src.FGSM.cleverhans.cleverhans.attacks.madry_et_al import MadryEtAl
from src.FGSM.cleverhans.cleverhans.attacks.max_confidence import MaxConfidence
from src.FGSM.cleverhans.cleverhans.attacks.noise import Noise
from src.FGSM.cleverhans.cleverhans.attacks.spsa import SPSA, projected_optimization
from src.FGSM.cleverhans.cleverhans.model import Model, CallableModelWrapper
from src.FGSM.cleverhans.cleverhans.compat import reduce_sum, reduce_mean
from src.FGSM.cleverhans.cleverhans.compat import reduce_max
from src.FGSM.cleverhans.cleverhans.compat import softmax_cross_entropy_with_logits
from src.FGSM.cleverhans.cleverhans.utils_tf import clip_eta

_logger = utils.create_logger("cleverhans.attacks")
tf_dtype = tf.as_dtype('float32')
