import numpy as np
import scipy as sp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load data

# Parameters
lr = 0.01			
gamma = 0.9			
batch_size = 128
lr_descent = 10

# Layers
'''
- 2 conv  layers
- 1 dense layer(no activation)
- 1 softmax layer
'''

# Convolution layer
def conv2d_basic(x, W, bias):
	conv = tf.layers.conv2d(x, W, kernel_size=(3,3), padding="SAME")
	return tf.nn.bias_add(conv, bias)

# Maxpool layer a 3x3 kernel and stride 3
def maxpool_3x3(x):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding="SAME") 

# Randon initialization of weights for convolution layers
def weight_variable(shape, stddev=0.2, name=None):
	initial = tf.truncated_normal(shape, stddev=stddev)
	if name is None:
		return tf.Variable(initial)
	return tf.get_variable(name, initializer = initial)

# Randon initialization of bias for convolution layers
def bias_variable(shape, name=None):
	initial = tf.constant(0.0, shape=shape)
	if name is None:
		return tf.Variable(initial)
	return tf.get_variable(name, initializer = initial)

# ReLU with nonlinearity generator
def NGReLu(x, t=-1):
	_x = x - tf.truncated_normal()

# assign variables


# run epochs
