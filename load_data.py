import os
import gzip
import shutil
import requests
import struct
import numpy as np
import matplotlib.pyplot as plt

saveIntermediate = False

def get_zipped_data(url, intermediate, out):
	r = requests.get(url, allow_redirects=True)
	if r.status_code == 200:
		open(intermediate, 'wb').write(r.content)
		with gzip.open(intermediate, 'rb') as f_in:
			with open(out, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
			if not saveIntermediate:
				os.remove(intermediate)

def download_data():
	data_location = os.path.join(os.curdir, 'data')
	training_X_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
	training_y_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
	test_X_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
	test_y_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
	if not os.path.exists(data_location):
		os.mkdir(data_location)
		get_zipped_data(training_X_url, 'data/training_X.gz', 'data/train-images.idx3-ubyte')
		get_zipped_data(training_y_url, 'data/training_y.gz', 'data/train-labels.idx1-ubyte')
		get_zipped_data(test_X_url, 'data/test_X.gz', 'data/t10k-images.idx3-ubyte')
		get_zipped_data(test_y_url, 'data/test_y.gz', 'data/t10k-labels.idx1-ubyte')

def read_data():
	with open('data/train-images.idx3-ubyte', 'rb') as train_X:
		endian, num_train_examples, num_rows, num_cols = struct.unpack('>IIII', train_X.read(16))
		training_images = np.fromfile(train_X, dtype=np.uint8)
	with open('data/train-labels.idx1-ubyte', 'rb') as train_y:
		struct.unpack(">II", train_y.read(8))
		training_labels = np.fromfile(train_y, dtype=np.uint8)
	with open('data/t10k-images.idx3-ubyte', 'rb') as test_X:
		endian, num_test_examples, num_rows, num_cols = struct.unpack(">IIII", test_X.read(16))
		test_images = np.fromfile(test_X, dtype=np.uint8)
	with open('data/t10k-labels.idx1-ubyte', 'rb') as test_y:
		struct.unpack(">II", test_y.read(8))
		test_labels = np.fromfile(test_y, dtype=np.uint8)

	training_images = training_images.reshape((num_train_examples, 784))
	test_images = test_images.reshape((num_test_examples, 784))

	return training_images, training_labels, test_images, test_labels

def show_image(index, images, labels):
	pixels = images[index].reshape((28, 28))
	plt.imshow(pixels, cmap = "gray_r")
	plt.xticks([])
	plt.yticks([])
	plt.title(labels[index])
	plt.show()