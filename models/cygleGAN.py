from __future__ import print_function, division
import scipy

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, LeakyReLU

from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, add