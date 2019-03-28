import tensorflow as tf


import numpy as np




targets = np.array([[2, 3, 4, 0]]).reshape(-1)
one_hot_targets = np.eye(10)[targets]

print(one_hot_targets)