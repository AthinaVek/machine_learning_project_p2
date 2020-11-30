import numpy as np
import keras
from keras import layers, optimizers, losses, metrics
from keras.models import Model, Sequential
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
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
import sys
from sklearn.metrics import classification_report
from classification_functions import MnistDataloader, print_plot, print_correct_incorrect


if __name__ == "__main__":
	if (len(sys.argv) == 11):
		i = 0
		for var in sys.argv:										# get values from command line
			if (var == "-d"):
				training_images_filepath = sys.argv[i + 1]
			if (var == "-dl"):
				training_labels_filepath = sys.argv[i + 1]
			if (var == "-t"):
				test_images_filepath = sys.argv[i + 1]
			if (var == "-tl"):
				test_labels_filepath = sys.argv[i + 1]
			if (var == "-model"):
				model = sys.argv[i + 1]
			i = i + 1
	else:
		print("Wrong input. Using default values.")
		training_images_filepath = 'train-images-idx3-ubyte'  		# default values if not given by user
		training_labels_filepath = 'train-labels-idx1-ubyte'
		test_images_filepath = 't10k-images-idx3-ubyte'
		test_labels_filepath = 't10k-labels-idx1-ubyte'
		model = 'autoencoder.h5'

	(xtrain, ytrain) = MnistDataloader(training_images_filepath, training_labels_filepath)	# read datasets
	(xtest, ytest) = MnistDataloader(test_images_filepath, test_labels_filepath)

	x_train = np.array(xtrain)
	x_test = np.array(xtest)
	y_train = np.array(ytrain)
	y_test = np.array(ytest)

	x_train = x_train.astype('float32') / 255.						# values 0/1
	x_test = x_test.astype('float32') / 255.

	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

	y_train_array = to_categorical(y_train)							# fix hot
	y_test_array = to_categorical(y_test)

	x_train, xx_test, y_train_array, xy_test_array = train_test_split(x_train, y_train_array, test_size=0.2, random_state=13)	#split dataset

	list_loss = []
	list_val_loss = []
	list_accuracy = []
	list_val_accuracy = []

	list_epochs_num = []
	list_batch_sz = []
	list_neurons_fc = []

	while (1):
		num_classes = 10

		epochs_num = int(input("GIVE NUMBER OF EPOCHS: \n"))						# get values from user
		batch_sz = int(input("GIVE BATCH SIZE: \n"))
		neurons_fc = int(input("GIVE NEURONS AT FC LAYER: \n"))

		# epochs_num = 5
		# batch_sz = 100
		# neurons_fc = 256

		list_epochs_num.append(epochs_num)
		list_batch_sz.append(batch_sz)
		list_neurons_fc.append(neurons_fc)

		networkInput = keras.layers.Input(shape=(28, 28, 1), name='input')

		autoencoder_model = load_model(model)										# load autoencoder model
		l = (len(autoencoder_model.layers)/2) -1

		flat = keras.layers.Flatten()(autoencoder_model.layers[int(l)].output)      # get autoencoder's output and do flatten
		den = keras.layers.Dense(neurons_fc, activation='relu')(flat)
		output = keras.layers.Dense(num_classes, activation='softmax')(den)

		encoder_model = Model(inputs=autoencoder_model.layers[0].output, outputs=output, name='ENCODER')

		print("\nStage 1:\n")														# stage 1: only fully connected layer
		for layer in encoder_model.layers[0:int(l)+1]:
			layer.trainable = False

		encoder_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
		encoder_model.summary()

		classify_train = encoder_model.fit(x_train, y_train_array, batch_size=batch_sz, epochs=epochs_num, verbose=1, validation_data=(xx_test, xy_test_array))

		print("\nStage 2:\n")														# stage 2: all layers
		for layer in encoder_model.layers[0:int(l)+1]:
			layer.trainable = True

		encoder_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
		
		classify_train = encoder_model.fit(x_train, y_train_array, batch_size=batch_sz, epochs=epochs_num, verbose=1, validation_data=(xx_test, xy_test_array))

		last_loss =  classify_train.history['loss'][-1]								# get last value to use for plots
		last_val_loss = classify_train.history['val_loss'][-1]
		last_accuracy = classify_train.history['accuracy'][-1]
		last_val_accuracy = classify_train.history['val_accuracy'][-1]

		list_loss.append(last_loss)
		list_val_loss.append(last_val_loss)
		list_accuracy.append(last_accuracy)
		list_val_accuracy.append(last_val_accuracy)

		val = int(input("TO REPEAT THE EXPERIMENT PRESS 1.\nTO SHOW THE PLOTS PRESS 2.\nTO CLASSIFY IMAGES PRESS 3.\n"))
		if (val == 1):
			continue

		elif (val == 2):
			val = int(input("TO PRINT EPOCHS PLOT PRESS 0.\nTO PRINT BATCH_SIZE PLOT PRESS 1.\nTO PRINT NEURONS_FC PLOT PRESS 2.\n"))
			print_plot(int(val), list_loss, list_val_loss, list_accuracy, list_val_accuracy, list_epochs_num, list_batch_sz, list_neurons_fc)	# print plots

			y_pred = encoder_model.predict(x_test, batch_size=batch_sz, verbose=1)
			y_pred_bool = np.argmax(y_pred, axis=1)
			print(classification_report(y_test, y_pred_bool))										# print report

			val = int(input("TO REPEAT THE EXPERIMENT PRESS 1.\nTO CLASSIFY IMAGES PRESS 3.\n"))
			if (val == 1):
				continue
			elif (val == 3):
				predicted_classes = encoder_model.predict(x_test)
				predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

				print_correct_incorrect(predicted_classes, x_test, y_test)							# print correct and incorrect images
				break

		elif (val == 3):
			predicted_classes = encoder_model.predict(x_test)
			predicted_classes = np.argmax(np.round(predicted_classes), axis=1)	

			print_correct_incorrect(predicted_classes, x_test, y_test)								# print correct and incorrect images

			break





