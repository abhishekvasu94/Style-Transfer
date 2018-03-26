import numpy as np
import tensorflow as tf
import cv2
import collections

class StyleTransfer:

	def __init__(self, content_img, style_img, noisy_img, alpha, beta, lr, n_epochs, session, model):

		"""
		content_img: This is the content image
		style_img: This is the style image
		noisy_img: This is the noisy image which subsequently gets trained to give a stylised output
		alpha: Weight for content
		beta: Weight for style
		lr: Learning rate
		n_epochs: Number of epochs
		sessions: The Tensorflow session
		model: The pretrained model (in this case VGG16)

		"""

		self.model = model

		self.content_img = content_img
		self.style_img = style_img
		self.noisy_img = noisy_img

		self.alpha = alpha
		self.beta = beta

		self.lr = lr
		self.n_epochs = n_epochs
		self.sess = session
		self.model = model

	def build(self):

		self.p = tf.placeholder(tf.float32, shape = self.content_img.shape, name = "p")
		self.a = tf.placeholder(tf.float32, shape = self.style_img.shape, name = "a")

		self.x = tf.Variable(self.noisy_img, trainable = True, dtype = tf.float32)

		CONTENT = ['conv4_2', 'conv5_2']						#Content layers
		STYLE = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']			#Style layers

		MIXED = ['conv4_2', 'conv5_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

		with tf.variable_scope("style"):

			style_vgg = self.model.build(self.p)
			self.style_layers = {}
			self.style_layers['conv1_1'] = self.gram_mat(self.model.conv1_1)
			self.style_layers['conv2_1'] = self.gram_mat(self.model.conv2_1)
			self.style_layers['conv3_1'] = self.gram_mat(self.model.conv3_1)
			self.style_layers['conv4_1'] = self.gram_mat(self.model.conv4_1)
			self.style_layers['conv5_1'] = self.gram_mat(self.model.conv5_1)
			
		with tf.variable_scope("content"):

			content_vgg = self.model.build(self.a)
			self.content_layers = {}
			self.content_layers['conv4_2'] = self.model.conv4_2
			self.content_layers['conv5_2'] = self.model.conv5_2

		with tf.variable_scope("mixed"):

			mixed_vgg = self.model.build(self.x)
			self.mixed_layers = {}
			self.mixed_layers['conv4_2'] = self.model.conv4_2
			self.mixed_layers['conv5_2'] = self.model.conv5_2
			self.mixed_layers['conv1_1'] = self.model.conv1_1
			self.mixed_layers['conv2_1'] = self.model.conv2_1
			self.mixed_layers['conv3_1'] = self.model.conv3_1
			self.mixed_layers['conv4_1'] = self.model.conv4_1
			self.mixed_layers['conv5_1'] = self.model.conv5_1

	#From this point, I need to calculate the loss with respect to style and content separately

		content_loss = 0	#Initialise content loss
		style_loss = 0		#Initialise style loss

		w = {'conv1_1': 0.5, 'conv2_1': 1.0, 'conv3_1': 1.5, 'conv4_1': 3.0, 'conv5_1': 4.0}

		for i in range(len(MIXED)):

			if MIXED[i] in self.content_layers.keys():

				P = self.content_layers[MIXED[i]]
				F = self.mixed_layers[MIXED[i]]

				content_loss += 0.5*tf.reduce_sum(tf.pow((F-P), 2))

			elif MIXED[i] in self.style_layers.keys():

				A = self.style_layers[MIXED[i]]
				temp = self.mixed_layers[MIXED[i]]

				_, m, n, l = temp.get_shape()	
				dims = m.value*n.value 
				M = l.value

				G = self.gram_mat(temp)
				#w = n.value

				style_loss += w[MIXED[i]]*(1.0/(4*(dims**2)*(M**2)))*tf.reduce_sum(tf.pow((A-G), 2))

		self.total_loss = self.alpha*content_loss + self.beta*style_loss

		train = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)	#The paper uses L-BFGS, but I am using Adam for convenience

		init = tf.global_variables_initializer()

		self.sess.run(init)

		for epoch in range(self.n_epochs):
			self.sess.run(train, feed_dict = {self.a: self.content_img, self.p: self.style_img})
			print "Epoch: {}".format(epoch)
			print "Loss: {}".format(self.sess.run(self.total_loss, feed_dict = {self.a: self.content_img, self.p: self.style_img}))

		computed_img = self.sess.run(self.x)

		final_img = self._return_img(computed_img)

		self.final_img = final_img

		return final_img

	def _return_img(self, img):

		MEAN = [123.68, 116.78, 103.94]

		for i in range(img.shape[-1]):
			img[:,:,:,i] += MEAN[i]

		return np.clip(img, 0.0, 255.0)


	def gram_mat(self, mat):

		num_channels = int(mat.get_shape()[-1])
		mat_reshape = tf.reshape(mat, [-1, num_channels])
		return tf.matmul(tf.transpose(mat_reshape), mat_reshape)
