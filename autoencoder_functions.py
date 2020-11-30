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
		images[i][:] = img

	return images


def encoder_decoder(x, layers, filters_size, filters_num):
	count = 0
	for l in range(layers):                                 # ENCODER (runs for given number of layers, filters_num, filters_size)
		conv_name = 'conv_' + str(l + 1)

		x = keras.layers.Conv2D(filters_num, kernel_size=(filters_size, filters_size), padding='same', activation='relu', name=conv_name)(x)
		x = keras.layers.BatchNormalization()(x)
		if (count < 3):
			x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
			count = count + 1
		# x = keras.layers.Dropout(0.5)(x)
		filters_num = filters_num * 2
	filters_num = filters_num / 2

	for l in range(layers - 1):                             # DECODER (runs for given number of layers, filters_num, filters_size)
		conv_name = 'conv_' + str(l + layers + 1)

		x = keras.layers.Conv2D(filters_num, kernel_size=(filters_size, filters_size), padding='same', activation='relu', name=conv_name)(x)
		x = keras.layers.BatchNormalization()(x)
		if (count - 1 > 0):
			x = keras.layers.UpSampling2D(size=(2, 2))(x)
			count = count - 1

		filters_num = filters_num / 2
	conv_name = 'conv_' + str(l + layers + 2)
	x = keras.layers.Conv2D(filters_num, kernel_size=(filters_size, filters_size), activation='relu', name=conv_name)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.UpSampling2D(size=(2, 2))(x)

	output = keras.layers.Conv2D(filters=1, kernel_size=(filters_size, filters_size), padding='same', activation='sigmoid', name='output')(x)

	return output


def print_digits(history):
	n = 10                              # number of classes
	plt.figure(figsize=(20, 4))
	for i in range(n):
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_test[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(out_images[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()


def print_plot(id, list_loss, list_val_loss, list_accuracy, list_val_accuracy, list_layers, list_filters_size,  list_filters_num, list_epochs_num, list_batch_sz ):
	y1 = list_loss
	y2 = list_val_loss
	y3 = list_accuracy
	y4 = list_val_accuracy

	plt.figure(figsize=(15, 4))
	plt.ylabel('loss')

	if id == 0:
		x = list_layers
		plt.title('layers')
		plt.xlabel('layers')
	elif id == 1:
		x = list_filters_size
		plt.title('filters_size')
		plt.xlabel('filters_size')
	elif id == 2:
		x = list_filters_num
		plt.title('filters_num')
		plt.xlabel('filters_num')
	elif id == 3:
		x = list_epochs_num
		plt.title('epochs_num')
		plt.xlabel('epochs_num')
	elif id == 4:
		x = list_batch_sz
		plt.title('batch_sz')
		plt.xlabel('batch_sz')

	plt.plot(x, y1)
	plt.plot(x, y2)
	plt.legend(['loss', 'val_loss'], loc='upper left')
	plt.show()

	plt.figure(figsize=(15, 4))
	plt.ylabel('accuracy')

	if id == 0:
		x = list_layers
		plt.title('layers')
		plt.xlabel('layers')
	elif id == 1:
		x = list_filters_size
		plt.title('filters_size')
		plt.xlabel('filters_size')
	elif id == 2:
		x = list_filters_num
		plt.title('filters_num')
		plt.xlabel('filters_num')
	elif id == 3:
		x = list_epochs_num
		plt.title('epochs_num')
		plt.xlabel('epochs_num')
	elif id == 4:
		x = list_batch_sz
		plt.title('batch_sz')
		plt.xlabel('batch_sz')

	plt.plot(x, y3)
	plt.plot(x, y4)
	plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
	plt.show()