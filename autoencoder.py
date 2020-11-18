import numpy as np
import keras
from keras import layers, optimizers, losses, metrics
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import struct
from array import array
from os.path  import join
import random
import tensorflow as tf
import pandas as pd
import os


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



if __name__ == "__main__":
	training_images_filepath = 'train-images-idx3-ubyte'
	training_labels_filepath = 'train-labels-idx1-ubyte'
	test_images_filepath = 't10k-images-idx3-ubyte'
	test_labels_filepath = 't10k-labels-idx1-ubyte'

	mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
	(xtrain, ytrain), (xtest, ytest) = mnist_dataloader.load_data()
	xtrain, xtest, trainground, validground = train_test_split(xtrain,xtrain,test_size=0.2,random_state=13)

	x_train = np.array(xtrain)
	x_test = np.array(xtest)
	train_ground = np.array(trainground)
	valid_ground = np.array(validground)

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	train_ground = train_ground.astype('float32') / 255.
	valid_ground = valid_ground.astype('float32') / 255.

	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
	train_ground = np.reshape(train_ground, (len(train_ground), 28, 28, 1))
	valid_ground = np.reshape(valid_ground, (len(valid_ground), 28, 28, 1))

	input = keras.layers.Input(shape=(28, 28, 1), name='input')
	x = keras.layers.Conv2D(8, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_1')(input)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)  #14x14x8
	x = keras.layers.Dropout(0.7)(x)
	x = keras.layers.Conv2D(16, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_2')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)  #7x7x16
	x = keras.layers.Dropout(0.7)(x)
	x = keras.layers.Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_3')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)  #4x4x32
	x = keras.layers.Dropout(0.7)(x)
	x = keras.layers.Conv2D(64, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_4')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_5')(x)
	x = keras.layers.BatchNormalization()(x)

	x = keras.layers.Conv2D(128, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_6')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv2D(64, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_7')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_8')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.UpSampling2D(size=(2,2))(x) #8x8x32
	x = keras.layers.Conv2D(16, kernel_size=(3,3), padding = 'same', activation='relu', name='conv_9')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.UpSampling2D(size=(2,2))(x) #16x16x16
	x = keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu', name='conv_10')(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.UpSampling2D(size=(2,2))(x) #28x28x8

	output = keras.layers.Convolution2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid', name='output')(x)

	model = Model(inputs=input, outputs=output, name='CAE')
	model.compile(optimizer=keras.optimizers.RMSprop(), loss='mean_squared_error', metrics=['accuracy'])
	model.summary()

	history = model.fit(x_test, valid_ground, batch_size=100, epochs=20, shuffle=True, verbose=1, validation_data=(x_test, valid_ground))
	out_images = model.predict(x_test)

	n = 10  # How many digits we will display
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# Display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_test[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# Display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(out_images[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

