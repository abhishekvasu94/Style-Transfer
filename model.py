import numpy as np
import tensorflow as tf
import os

VGG_MEAN = [123.68, 116.78, 103.94]

class VGG():

	def __init__(self, path = None):

		try:
			self.data = np.load(path, encoding = 'latin1').item()

		except IOError:
			print "Please input the correct file"

	def build(self, img):


		img = tf.stack(axis = -1, values = [img[:,:,:,0] - VGG_MEAN[0], img[:,:,:,1] - VGG_MEAN[1], img[:,:,:,2] - VGG_MEAN[2]])

		self.conv1_1 = self.conv_layer(img, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		self.pool1 = self.avg_pool(self.conv1_2, "pool1")		#Max pool layer substituted with avg pool. It gives better results

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
		self.pool2 = self.avg_pool(self.conv2_2, "pool2")

		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
		self.pool3 = self.avg_pool(self.conv3_3, 'pool3')

		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
		self.pool4 = self.avg_pool(self.conv4_3, 'pool4')

		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
		self.pool5 = self.avg_pool(self.conv5_3, 'pool5')

		#The fully connected layers are not required, because they will cause an error for any input image that isn't (224,224,3). 

		#self.fc6 = self.fully_connected(self.pool5, "fc6")
		#self.relu6 = tf.nn.relu(self.fc6)

		#self.fc7 = self.fully_connected(self.relu6, "fc7")
		#self.relu7 = tf.nn.relu(self.fc7)

		#self.fc8 = self.fully_connected(self.relu7, "fc8")
		#self.prob = tf.nn.softmax(self.fc8, name="prob")

		print "Finished"


	def conv_layer(self, prev, layer_name):

		with tf.variable_scope(layer_name):

			filt = tf.constant(self.data[layer_name][0], name = 'conv_filter')
			convolution = tf.nn.conv2d(prev, filt, strides = [1,1,1,1], padding = "SAME")
			biases = tf.constant(self.data[layer_name][1], name = 'bias')
			adding_biases = tf.nn.bias_add(convolution, biases)
			relu_activation = tf.nn.relu(adding_biases)
			return relu_activation

	def fully_connected(self, prev, layer_name):

		with tf.variable_scope(layer_name):

			shape = prev.get_shape().as_list()
			dim = 1
			for d in shape[1:]:
				dim *= d
			flat_mat = tf.reshape(prev, [-1, dim])
			weights = tf.constant(self.data[layer_name][0], name = 'fully_connected_weights')
			biases = tf.constant(self.data[layer_name][1], name = 'fully_connected_biases')

			return tf.nn.bias_add(tf.matmul(flat_mat, weights), biases)

	def max_pool(self, prev, layer_name):

		return tf.nn.max_pool(prev, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = layer_name)

	def avg_pool(self, prev, layer_name):

		return tf.nn.avg_pool(prev, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = layer_name)
