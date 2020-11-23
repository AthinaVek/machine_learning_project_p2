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
import sys
from tensorflow.keras.models import load_model
import h5py


def MnistDataloader(training_images_filepath):
	with open(training_images_filepath, 'rb') as file:
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
	
	return images
	

def encoder_decoder(x, layers, filters_size, filters_num):
	count = 0
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


	for l in range(layers-1):							# DECODER
		conv_name = 'conv_' + str(l+layers+1)

		x = keras.layers.Conv2D(filters_num, kernel_size=(filters_size,filters_size), padding = 'same', activation='relu', name=conv_name)(x)
		x = keras.layers.BatchNormalization()(x)
		if (count-1 > 0):
			x = keras.layers.UpSampling2D(size=(2,2))(x)
			count = count-1

		filters_num = filters_num/2
	conv_name = 'conv_' + str(l+layers+2)
	x = keras.layers.Conv2D(filters_num, kernel_size=(filters_size,filters_size), activation='relu', name=conv_name)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.UpSampling2D(size=(2,2))(x)
	output = keras.layers.Conv2D(filters=1, kernel_size=(filters_size,filters_size), padding='same', activation='sigmoid', name='output')(x)

	return output


def print_plots(history):
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
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


if __name__ == "__main__":
	if (len(sys.argv) == 3):
		training_images_filepath = sys.argv[2]
	else:
		print ("Wrong input. Using default value.")
		training_images_filepath = 'train-images-idx3-ubyte'					# default train dataset
	
	(xtrain) = MnistDataloader(training_images_filepath)				# read images
	xtrain, xtest, trainground, validground = train_test_split(xtrain,xtrain,test_size=0.2,random_state=13)		# split dataset

	x_train = np.array(xtrain)
	x_test = np.array(xtest)
	train_ground = np.array(trainground)
	valid_ground = np.array(validground)

	x_train = x_train.astype('float32') / 255.									# values 0/1
	x_test = x_test.astype('float32') / 255.
	train_ground = train_ground.astype('float32') / 255.
	valid_ground = valid_ground.astype('float32') / 255.

	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
	train_ground = np.reshape(train_ground, (len(train_ground), 28, 28, 1))
	valid_ground = np.reshape(valid_ground, (len(valid_ground), 28, 28, 1))

	while (1):
		layers = int(input("GIVE NUMBER OF LAYERS: \n"))
		filters_size = int(input("GIVE FILTER SIZE: \n"))
		filters_num = int(input("GIVE NUMBER OF FILTERS IN FIRST LAYER: \n"))
		epochs_num = int(input("GIVE NUMBER OF EPOCHS: \n"))
		batch_sz = int(input("GIVE BATCH SIZE: \n"))
		
		# layers = 5
		# filters_size = 3
		# filters_num = 8
		# epochs_num = 10
		# batch_sz = 100
		
		networkInput = keras.layers.Input(shape=(28, 28, 1), name='input')
		x = networkInput

		output = encoder_decoder(x, layers, filters_size, filters_num)

		model = Model(inputs=networkInput, outputs=output, name='AUTOENCODER')
		model.compile(optimizer=keras.optimizers.RMSprop(), loss='mean_squared_error', metrics=['accuracy'])
		model.summary()

		history = model.fit(x_test, valid_ground, batch_size=batch_sz, epochs=epochs_num, shuffle=True, verbose=1, validation_data=(x_test, valid_ground))
		out_images = model.predict(x_test)

		val = int(input("TO REPEAT THE EXPERIMENT PRESS 1.\nTO SHOW THE PLOTS PRESS 2.\nTO SAVE THE MODEL PRESS 3.\n"))
		if(val == 1):
			continue

		elif(val == 2):
			print_plots(history)
			val = int(input("TO REPEAT THE EXPERIMENT PRESS 1.\nTO SAVE THE MODEL PRESS 3.\n"))
			if(val == 1):
				continue
			elif(val == 3):
				model.save('autoencoder_model', save_format='h5')
				break

		elif(val == 3):
			model.save('autoencoder_model', save_format='h5')
			break
