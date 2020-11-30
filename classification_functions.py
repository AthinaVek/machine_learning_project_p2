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


def MnistDataloader(images_filepath, labels_filepath):			# read images and labels from datasets
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
		images[i][:] = img

	return images, labels


def print_plot(id, list_loss, list_val_loss, list_accuracy, list_val_accuracy, list_epochs_num, list_batch_sz, list_neurons_fc):
	y1 = list_loss
	y2 = list_val_loss
	y3 = list_accuracy
	y4 = list_val_accuracy

	plt.subplot(2, 1, 1)
	plt.ylabel('loss')

	if id == 0:
		x = list_epochs_num
		plt.title('epochs_num')
		plt.xlabel('epochs_num')
	elif id == 1:
		x = list_batch_sz
		plt.title('batch_sz')
		plt.xlabel('batch_sz')
	elif id == 2:
		x = list_neurons_fc
		plt.title('neurons_fc')
		plt.xlabel('neurons_fc')

	plt.plot(x, y1)
	plt.plot(x, y2)
	plt.legend(['loss', 'val_loss'], loc='upper left')

	plt.subplot(2, 1, 2)
	plt.ylabel('accuracy')

	if id == 0:
		x = list_epochs_num
		plt.xlabel('epochs_num')
	elif id == 1:
		x = list_batch_sz
		plt.xlabel('batch_sz')
	elif id == 2:
		x = list_neurons_fc
		plt.xlabel('neurons_fc')

	plt.plot(x, y3)
	plt.plot(x, y4)
	plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
	plt.show()


def print_correct_incorrect(predicted_classes, x_test, y_test):
	correct = np.where(predicted_classes == y_test)[0]			# find correct predictions
	print ("\nCORRECT:")
	print(len(correct))

	i=1
	plt.figure(figsize=(15, 15))
	for correct in correct[:9]:
		ax = plt.subplot(3, 3, i)
		plt.imshow(x_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
		ax.set_title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		i = i+1
	plt.show()

	incorrect = np.where(predicted_classes != y_test)[0]		# find incorrect predictions
	print ("\nINCORRECT:")
	print(len(incorrect))

	plt.figure(figsize=(15, 15))
	for i, incorrect in enumerate(incorrect[:9]):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
		plt.gray()
		ax.set_title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[incorrect]))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()
