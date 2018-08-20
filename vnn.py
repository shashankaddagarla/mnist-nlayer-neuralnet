import numpy as np
import math
import load_data

N_CLASSES = 10
N_FEATURES = 28*28
RANDOM_SEED = 42

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))

class NNClassifier:

	def __init__(self, n_classes, n_features, layers=list(), l1 = 0.0, l2 = 0.0,
				epochs = 500, learning_rate = 1e-1, n_batches = 1):

		np.random.seed(RANDOM_SEED)
		self.n_classes = n_classes # number of neurons in output layer
		self.n_features = n_features # number of neurons in input layer
		self.l1 = l1
		self.l2 = l2
		self.layers = layers
		self.n_layers = len(layers)
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.n_batches = n_batches
		self.weights = self._init_weights()
		self.biases = [np.zeros((x, 1)) for x in self.layers]
		self.activations = [np.zeros((x, 1)) for x in self.layers]

	def _init_weights(self):
		weights = [np.random.uniform(-1.0, 1.0, (x, y)) for x, y in zip(self.layers[1:], self.layers[:-1])]
		weights.insert(0, np.array([0]))
		return weights

	def _forward_propagation(self, x):
		# we only want a single x because we're going to be doing stochastic gradient descent, so we'll be propagating training examples one-by-one
		self.activations[0] = x
		for i in range(1, self.n_layers):
			test = np.matmul(self.weights[i], self.activations[i-1]) + self.biases[i]
			self.activations[i] = sigmoid(test)
		print(self.activations[2])

	def _backward_propagation(self, x, y):
		pass


	def fit(self, X):
		self.error = []
		self._forward_propagation(X)




load_data.download_data()
training_images, training_labels, test_images, test_labels = load_data.read_data()

nn = NNClassifier(n_classes=N_CLASSES,
                  n_features=N_FEATURES,
                  layers = [N_FEATURES, 30, 10],
                  l1 = 0.0,
                  l2 = 0.5,
                  epochs = 300,
                  learning_rate = 0.001,
                  n_batches = 25)

example = training_images[0]
example = example.reshape(-1, 1)
nn.fit(example)



