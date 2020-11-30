import numpy as np
import keras
from keras import layers, optimizers, losses, metrics
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import struct
from array import array
from os.path import join
import random
import tensorflow as tf
import pandas as pd
import os
import sys
from tensorflow.keras.models import load_model
import h5py
from autoencoder_functions import MnistDataloader, encoder_decoder, print_plot


if __name__ == "__main__":
	if (len(sys.argv) == 3):                                    # get value from command line
		training_images_filepath = sys.argv[2]
	else:
		print("Wrong input. Using default value.")
		training_images_filepath = 'train-images-idx3-ubyte'    # default train dataset

	(xtrain) = MnistDataloader(training_images_filepath)        # read images

	xtrain, xtest, trainground, validground = train_test_split(xtrain, xtrain, test_size=0.2, random_state=13)      # split dataset

	x_train = np.array(xtrain)
	x_test = np.array(xtest)
	train_ground = np.array(trainground)
	valid_ground = np.array(validground)

	x_train = x_train.astype('float32') / 255.  				# values 0/1
	x_test = x_test.astype('float32') / 255.
	train_ground = train_ground.astype('float32') / 255.
	valid_ground = valid_ground.astype('float32') / 255.

	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
	train_ground = np.reshape(train_ground, (len(train_ground), 28, 28, 1))
	valid_ground = np.reshape(valid_ground, (len(valid_ground), 28, 28, 1))

	list_loss = []
	list_val_loss = []
	list_accuracy = []
	list_val_accuracy = []

	list_layers = []
	list_filters_size = []
	list_filters_num = []
	list_epochs_num = []
	list_batch_sz = []

	while (1):
		layers = int(input("GIVE NUMBER OF LAYERS: \n"))                            # get values from user
		filters_size = int(input("GIVE FILTER SIZE: \n"))
		filters_num = int(input("GIVE NUMBER OF FILTERS IN FIRST LAYER: \n"))
		epochs_num = int(input("GIVE NUMBER OF EPOCHS: \n"))
		batch_sz = int(input("GIVE BATCH SIZE: \n"))

		# layers = 3
		# filters_size = 3
		# filters_num = 8
		# epochs_num = 10
		# batch_sz = 64

		list_layers.append(layers)
		list_filters_size.append(filters_size)
		list_filters_num.append(filters_num)
		list_epochs_num.append(epochs_num)
		list_batch_sz.append(batch_sz)

		networkInput = keras.layers.Input(shape=(28, 28, 1), name='input')
		x = networkInput

		output = encoder_decoder(x, layers, filters_size, filters_num)              # call encoder and decoder

		model = Model(inputs=networkInput, outputs=output, name='AUTOENCODER')
		model.compile(optimizer=keras.optimizers.RMSprop(), loss='mean_squared_error', metrics=['accuracy'])
		model.summary()

		history = model.fit(x_train, train_ground, batch_size=batch_sz, epochs=epochs_num, shuffle=True, verbose=1, validation_data=(x_test, valid_ground))     # fit

		last_loss =  history.history['loss'][-1]                                    # get last value to use for plots
		last_val_loss = history.history['val_loss'][-1]
		last_accuracy = history.history['accuracy'][-1]
		last_val_accuracy = history.history['val_accuracy'][-1]

		list_loss.append(last_loss)
		list_val_loss.append(last_val_loss)
		list_accuracy.append(last_accuracy)
		list_val_accuracy.append(last_val_accuracy)

		out_images = model.predict(x_test)                                          # predict

		val = int(input("TO REPEAT THE EXPERIMENT PRESS 1.\nTO SHOW THE PLOTS PRESS 2.\nTO SAVE THE MODEL PRESS 3.\n"))
		if (val == 1):
			continue

		elif (val == 2):
			val = int(input("TO PRINT LAYERS PLOT PRESS 0.\nTO PRINT FILTERS_SIZE PLOT PRESS 1.\nTO PRINT FILTERS_NUM PLOT PRESS 2.\nTO PRINT EPOCHS PLOT PRESS 3.\nTO PRINT BATCH_SIZE PLOT PRESS 4.\n"))
			print_plot(int(val), list_loss, list_val_loss, list_accuracy, list_val_accuracy, list_layers, list_filters_size,  list_filters_num, list_epochs_num, list_batch_sz)     # print plots

			val = int(input("TO REPEAT THE EXPERIMENT PRESS 1.\nTO SAVE THE MODEL PRESS 3.\n"))
			if (val == 1):
				continue
			elif (val == 3):
				filename = input("Type the filename for the model: ")
				model.save(filename)        # save model
				break

		elif (val == 3):
			filename = input("Type the filename for the model: ")
			model.save(filename)            # save model
			break