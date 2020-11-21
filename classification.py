import numpy as np
import keras
from keras import layers, optimizers, losses, metrics
from keras.models import Model,Sequential
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
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from tensorflow.keras.models import load_model


class MnistDataloader(object):
	def __init__(self, images_filepath,labels_filepath):
		self.images_filepath = images_filepath
		self.labels_filepath = labels_filepath
	
	def read_images_labels(self):        
		labels = []
		with open(self.labels_filepath, 'rb') as file:
			magic, size = struct.unpack(">II", file.read(8))
			if magic != 2049:
				raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
			labels = array("B", file.read())        
		
		with open(self.images_filepath, 'rb') as file:
			magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
			if magic != 2051:
				raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
			image_data = array("B", file.read())        
		images = []
		for i in range(size):
			images.append([0] * rows * cols)
		for i in range(size):
			img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
			images[i][:] = img            
		
		return images, labels
	

if __name__ == "__main__":
	if (len(sys.argv) == 11):
		i = 0
		for var in sys.argv:
			if (var == "-d"):
				training_images_filepath = sys.argv[i+1]
			if (var == "-dl"):
				training_labels_filepath = sys.argv[i+1]
			if (var == "-t"):
				test_images_filepath = sys.argv[i+1]
			if (var == "-tl"):
				test_labels_filepath = sys.argv[i+1]
			if (var == "-model"):
				model = sys.argv[i+1]
			i = i+1
	else:
		print ("AAAAAAAAAAAAAAAAAAAA")
		training_images_filepath = 'train-images-idx3-ubyte'					# default values if not given by user
		training_labels_filepath = 'train-labels-idx1-ubyte'
		test_images_filepath = 't10k-images-idx3-ubyte'
		test_labels_filepath = 't10k-labels-idx1-ubyte'

	mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath)
	(xtrain, ytrain)= mnist_dataloader.read_images_labels()

	mnist_dataloader = MnistDataloader(test_images_filepath, test_labels_filepath)
	(xtest, ytest) = mnist_dataloader.read_images_labels()

	x_train = np.array(xtrain)
	x_test = np.array(xtest)
	y_train = np.array(ytrain)
	y_test = np.array(ytest)

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.

	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

	y_train_array = to_categorical(y_train)
	y_test_array = to_categorical(y_test)

	x_train,xx_test,y_train_array,xy_test_array = train_test_split(x_train,y_train_array,test_size=0.2,random_state=13)
	
	layers = 5
	filters_size = 3
	filters_num = 8
	epochs_num = 5
	batch_sz = 100
	num_classes = 10

	count = 0
	networkInput = keras.layers.Input(shape=(28, 28, 1), name='input')
	x = networkInput

	for l in range(layers):							# ENCODER
		conv_name = 'conv_' + str(l+1)

		x = keras.layers.Conv2D(filters_num, kernel_size=(filters_size,filters_size), padding = 'same', activation='relu', name=conv_name)(x)
		x = keras.layers.BatchNormalization()(x)
		if (count < 3):
			x = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
			count = count+1
		x = keras.layers.Dropout(0.5)(x)
		filters_num = filters_num*2
	filters_num = filters_num/2

	flat = keras.layers.Flatten()(x)
	den = keras.layers.Dense(filters_num, activation='relu')(flat)
	output = keras.layers.Dense(num_classes, activation='softmax')(den)

	encoder_model = Model(inputs=networkInput, outputs=output, name='ENCODER')

	autoencoder_model = load_model(model)
	for l1,l2 in zip(encoder_model.layers[:19],autoencoder_model.layers[0:19]):
		l1.set_weights(l2.get_weights())

	autoencoder_model.get_weights()[0][1]
	encoder_model.get_weights()[0][1]

	for layer in encoder_model.layers[0:19]:
		layer.trainable = False

	encoder_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	encoder_model.summary()

	classify_train = encoder_model.fit(x_train, y_train_array, batch_size=batch_sz, epochs=epochs_num, verbose=1, validation_data=(xx_test, xy_test_array))

	encoder_model.save_weights('autoencoder_classification.h5')

	for layer in encoder_model.layers[0:19]:
		layer.trainable = True

	encoder_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

	classify_train = encoder_model.fit(x_train, y_train_array, batch_size=batch_sz, epochs=epochs_num, verbose=1, validation_data=(xx_test, xy_test_array))

	encoder_model.save_weights('classification_complete.h5')

	predicted_classes = encoder_model.predict(x_test)
	predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

	correct = np.where(predicted_classes==y_test)[0]
	print (len(correct))
	for i, correct in enumerate(correct[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(x_train[correct].reshape(28,28), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
		plt.tight_layout()