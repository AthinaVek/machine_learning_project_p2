import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import tensorflow as tf
import pandas as pd
import os
import keras
from keras.models import load_model
from keras import layers, optimizers, losses, metrics


class MnistDataloader(object):
	def __init__(self, training_images_filepath,training_labels_filepath, test_images_filepath, test_labels_filepath):
		self.training_images_filepath = training_images_filepath
		self.training_labels_filepath = training_labels_filepath
		self.test_images_filepath = test_images_filepath
		self.test_labels_filepath = test_labels_filepath
	
	def read_images_labels(self, images_filepath, labels_filepath):        
		labels = []
		with open(labels_filepath, 'rb') as file:
			magic, size = struct.unpack(">II", file.read(8))
			if magic != 2049:
				raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
			labels = array("B", file.read())        
		
		with open(images_filepath, 'rb') as file:
			magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
			if magic != 2051:
				raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
			image_data = array("B", file.read())        
		images = []
		for i in range(size):
			images.append([0] * rows * cols)
		for i in range(size):
			img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
			# img = img.reshape(28, 28)
			images[i][:] = img            
		
		return images, labels
			
	def load_data(self):
		x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
		x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
		return (x_train, y_train), (x_test, y_test)


def encoder(input_img):
	#encoder
	#input = 28 x 28 x 1 (wide and thin)

	input_img = keras.Input(shape=(28,28,1))
	conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
	conv1 = tf.keras.layers.BatchNormalization()(conv1)
	conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = tf.keras.layers.BatchNormalization()(conv1)
	pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
	conv2 = tf.keras.layers.BatchNormalization()(conv2)
	conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = tf.keras.layers.BatchNormalization()(conv2)
	pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small & thick)
	conv3 = tf.keras.layers.BatchNormalization()(conv3)
	conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = tf.keras.layers.BatchNormalization()(conv3)
	conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small & thick)
	conv4 = tf.keras.layers.BatchNormalization()(conv4)
	conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = tf.keras.layers.BatchNormalization()(conv4)
	print (conv4)
	return conv4


if __name__ == "__main__":
	training_images_filepath = 'train-images-idx3-ubyte'
	training_labels_filepath = 'train-labels-idx1-ubyte'
	test_images_filepath = 't10k-images-idx3-ubyte'
	test_labels_filepath = 't10k-labels-idx1-ubyte'

	mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
	(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

	encoder(x_train[0])

	# print(x_train[0])
	# print(y_train[0])












# images_2_show = []
# titles_2_show = []
# for i in range(0, 10):
#     r = random.randint(1, 60000)
#     images_2_show.append(x_train[r])
#     titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

# for i in range(0, 5):
#     r = random.randint(1, 10000)
#     images_2_show.append(x_test[r])        
#     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

