#! /usr/bin/env python3

import tensorflow as tf
import io
import PIL
from PIL import Image
import numpy
import os

import settings

class NeuralNetwork:

	def __init__(self):
		self.getNetworkModel()

	def getNetworkModel(self):

		# Number of nodes in a specific layer
		numNodesImp = 784

		numNodesHl1 = 2000
		numNodesHl2 = 1000
		numNodesHl3 = 500

		numNodesOut = 10

		# Import node
		self.x = tf.placeholder("float")

		self.hl1 = {"weights": tf.Variable(tf.random_normal([numNodesImp, numNodesHl1]), name="hl1w"),
				"biases": tf.Variable(tf.random_normal([numNodesHl1]), name="hl1b")}

		self.hl2 = {"weights": tf.Variable(tf.random_normal([numNodesHl1, numNodesHl2]), name="hl2w"),
				"biases": tf.Variable(tf.random_normal([numNodesHl2]), name="hl2b")}

		self.hl3 = {"weights": tf.Variable(tf.random_normal([numNodesHl2, numNodesHl3]), name="hl3w"),
				"biases": tf.Variable(tf.random_normal([numNodesHl3]), name="hl3b")}

		self.out = {"weights": tf.Variable(tf.random_normal([numNodesHl3, numNodesOut]), name="outw"),
		"biases": tf.Variable(tf.random_normal([numNodesOut]), name="outb")}

	def feedToNetwork(self, data):

		print("Evaluating input in cnn")

		l1 = tf.add(tf.matmul(data, self.hl1["weights"]), self.hl1["biases"])
		l1 = tf.nn.relu(l1)

		l2 = tf.add(tf.matmul(l1, self.hl2["weights"]), self.hl2["biases"])
		l2 = tf.nn.relu(l2)

		l3 = tf.add(tf.matmul(l2, self.hl3["weights"]), self.hl3["biases"])
		l3 = tf.nn.relu(l3)

		self.output = tf.add(tf.matmul(l3, self.out["weights"]), self.out["biases"])

		return self.output

	def trainNetwork(self):

		print("Train neural network")

		from tensorflow.examples.tutorials.mnist import input_data

		mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

		y = tf.placeholder("float")
		batch_size = 100

		currentOut = self.feedToNetwork(self.x)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=currentOut, labels=y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		numEpochs = 10
		batchSize = 100

		self.saver = tf.train.Saver({"hl1W": self.hl1["weights"], "hl1B": self.hl1["biases"], "hl2W": self.hl2["weights"], "hl2B": self.hl2["biases"], "hl3W": self.hl3["weights"], "hl3B": self.hl3["biases"], "outw": self.out["weights"], "outb": self.out["biases"]})

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			for epoch in range(numEpochs):
				epochLoss = 0
				for _ in range(int(mnist.train.num_examples / batchSize)):
					epochX, epochY = mnist.train.next_batch(batchSize)

					_, c = sess.run([optimizer, cost], feed_dict={self.x: epochX, y: epochY})
					epochLoss += c

				print("Epoch", epoch, "completed out of", numEpochs, "loss:", epochLoss)

			correct = tf.equal(tf.argmax(currentOut, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, "float"))
			print("Accuracy:", accuracy.eval({self.x: mnist.test.images, y: mnist.test.labels}))

			# Save cnn
			print("Saveing network")

			if not os.path.exists(settings.saveFilePath):
				os.makedirs(settings.saveFilePath)

			settings.saveFilePath = self.saver.save(sess, settings.saveFilePath)
			print("Saved model to ", settings.saveFilePath)

	def imageToMnist(self, image):
		"""
			Loads the image, downscales it to 28x28px, converts it to a numpy rgb array, subtracts it from 1 in order to make 0 = white, 1 = black, converts it to a grayscale double flattened array
		"""
		side = (20, 20)
		image = image
		image.thumbnail(side, PIL.Image.ANTIALIAS)

		rbgImageArray = numpy.asarray(image)
		imageArray = []


		for row in rbgImageArray:
			for column in row:
				imageArray.append(1 - float(column[0] / 255))

		# Make a 4px wide border around the image
		fullImage = []

		for i in range(4*28):
			fullImage.append(0.0)

		for i in range(20):

			for j in range(4):
				fullImage.append(0.0)

			for _ in range(20):
				fullImage.append(imageArray[0])
				imageArray.pop(0)

			for j in range(4):
				fullImage.append(0.0)

		for i in range(4*28):
			fullImage.append(0.0)

		return fullImage

	def _displayArrayImage(self, binary):
		binaryArray = binary

		for i in range(len(binaryArray)):
			binaryArray[i] = (1 - binaryArray[i]) * 255

		print("disp", binaryArray)

		img = Image.new('L', (28, 28))
		img.putdata(binary)
		img.show()


