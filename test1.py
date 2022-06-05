

import numpy as np
import tensorflow as tf

data = np.array([[6.0, 2.0], [2.0, 3.0]])
values, vectors = tf.linalg.eigh(data)
print('eigenvector :',vectors)
print('eigenvalues :',values)


