import numpy as np
import tensorflow as tf
from model import VGG
import cv2
import pdb
from style_transfer import StyleTransfer

def resize_img(img, HEIGHT, WIDTH):
	res_img = np.zeros((HEIGHT,WIDTH,3))
	for i in range(img.shape[-1]):
		res_img[:,:,i] = cv2.resize(img[:,:,i], (WIDTH, HEIGHT))
	res_img = np.expand_dims(res_img, axis = 0)
	res_img  = np.float32(res_img)
	return res_img


if __name__ == "__main__":

	cont = cv2.imread("profile_pic.jpg", 1)
	st = cv2.imread("starry-night.jpg", 1)
	path = "vgg16.npy"
	HEIGHT = 600
	WIDTH = 800

	content_img = resize_img(cont, HEIGHT, WIDTH)
	stylized_img = resize_img(st, HEIGHT, WIDTH)
	noisy_img = np.random.normal(scale = np.std(content_img), size = content_img.shape)

	noisy_img = np.float32(noisy_img)

	vgg16 = VGG(path)

	#Play around with these parameters
	alpha = 5e-4
	beta = 1	

	with tf.Session() as sess:

		init = tf.global_variables_initializer()
		sess.run(init)

		st_transfer = StyleTransfer(content_img, stylized_img, noisy_img, alpha, beta, lr = 5, n_epochs = 2000, session = sess, model = vgg16)

		st_transfer.build()
		final_img = st_transfer.final_img

	sess.close()


	cv2.imwrite('final_img.jpg', final_img[0])
