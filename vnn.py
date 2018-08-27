import os
import random
import math

import numpy as np
import load_data


N_CLASSES = 10
N_FEATURES = 28*28
RANDOM_SEED = 42

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def one_hot(y):
	array = np.zeros((N_CLASSES, 1))
	if 0 <= y < 10:
		array[y] = 1
	return array

class NNClassifier:

	def __init__(self, n_features, layers=list(), l2 = 0.0,
				epochs = 500, learning_rate = 1e-1):

		np.random.seed(RANDOM_SEED)
		random.seed(RANDOM_SEED)
		self.n_features = n_features # number of neurons in input layer
		self.l2 = l2
		self.layers = layers
		self.n_layers = len(layers)
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.weights = self._init_weights()
		self.biases = [np.random.randn(x, 1) for x in self.layers]
		self.activations = [np.zeros((x, 1)) for x in self.layers]

	def _init_weights(self):
		# in testing, uniform initialization prevented vanishing gradient problem better than standard normal distribution with mean 0 and stddev 1
		weights = [np.random.randn(x, y)/np.sqrt(y) for x, y in zip(self.layers[1:], self.layers[:-1])]
		weights.insert(0, np.array([0]))
		return weights

	def _forward_propagation(self, x):
		# we only want a single x because we're going to be doing stochastic gradient descent, so we'll be propagating training examples one-by-one
		self.activations[0] = x
		for i in range(1, self.n_layers):
			test = np.matmul(self.weights[i], self.activations[i-1]) + self.biases[i]
			self.activations[i] = sigmoid(test)

	def _backward_propagation(self, x, y):
		# delta_l represents layer-wise error
		delta_l = [np.zeros((x, 1)) for x in (self.layers[1:])]
		delta_l.insert(0, np.array([0]))
		delta_l[-1] = self.activations[-1] - y # cross-entropy error delta

		for layer in range(self.n_layers - 2, 0, -1):
			sigmoid_prime_z = np.multiply(self.activations[layer], 1-self.activations[layer])
			delta_l[layer] = np.multiply(np.matmul(self.weights[layer + 1].T, delta_l[layer + 1]), sigmoid_prime_z)

		return delta_l

	def fit(self, training_images, training_labels, test_images, test_labels):

		X = np.split(training_images, training_images.shape[0])
		y = np.split(training_labels, training_labels.shape[0])

		data = list(zip(X, y))
		training_data = data[:-500]
		validation_data = data[-500:]

		for epoch in range(self.epochs):
			random.shuffle(training_data)

			for x, y in data:
				x = x.reshape((1, -1)).T
				y = one_hot(y)
				self._forward_propagation(x)
				delta_l = self._backward_propagation(x, y)

				for layer in range(self.n_layers - 1, 0, -1):
					self.weights[layer] -= self.learning_rate * (((self.l2 / len(training_data)) * self.weights[layer]) + (np.matmul(delta_l[layer], self.activations[layer - 1].T)))
					self.biases[layer] -= self.learning_rate * delta_l[layer]

			print("epoch {} gets {} correct out of {}".format(epoch, self.validation_accuracy(validation_data), len(validation_data)))

		print("training set accuracy: {}%".format(self.validation_accuracy(training_data) / len(training_data)))
		print("validation set accuracy: {}%".format(self.validation_accuracy(validation_data) / len(validation_data)))

		X_test = np.split(test_images, test_images.shape[0])
		y_test = np.split(test_labels, test_labels.shape[0])
		test_data = list(zip(test_images, test_labels))
		print("test set accuracy: {}%".format(self.validation_accuracy(test_data) / len(test_data)))

	def predict(self, training_example):
		self._forward_propagation(training_example)
		return np.argmax(self.activations[-1])

	def validation_accuracy(self, validation_data):
		num_correct = 0
		for x, y in validation_data:
			if self.predict(x.reshape((1, -1)).T) == y:
				num_correct += 1;
		return num_correct

	def save_model(self, filename='model.npz'):
		np.savez_compressed(
			file=os.path.join(os.curdir, 'models', filename),
			weights=self.weights,
			biases=self.biases,
			layers=self.layers,
			epochs=self.epochs,
			learning_rate=self.learning_rate
		)

	def load_model(self, filename='model.npz'):
		model = np.load(os.path.join(os.curdir, 'models', filename))
		self.weights = model['weights']
		self.biases = model['biases']
		self.layers = model['layers']
		self.num_layers = len(layers)
		self.epochs = model['epochs']
		self.learning_rate = model['learning_rate']
		self.activations = [np.zeros((x, 1)) for x in self.layers]

load_data.download_data()
training_images, training_labels, test_images, test_labels = load_data.read_data()

nn = NNClassifier(n_features=N_FEATURES,
                  layers = [N_FEATURES, 30, 10],
                  l2 = 0.5,
                  epochs = 30,
                  learning_rate = 0.001)

nn.fit(training_images, training_labels, test_images, test_labels)



