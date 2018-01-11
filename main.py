#! /usr/bin/env python3

from tkinter import *		# GUI
import io 					# Get Canvas Content
from PIL import Image
import PIL
import numpy

import tensorflow as tf 	# Neural network

import sys					# Exiting
import subprocess			# Clear Terminal

root = None
image = None

neuralNetwork = None

class Window(Frame):

	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.master = master

		self.master.title("Draw a Number")
		self.pack(fill=BOTH, expand=1)

		self.addCanvas()
		self.addBottons()


	def addCanvas(self):
		self.canvas = Canvas(self.master, width=500, height=500, bg="gray")
		self.canvas.pack()

		self.canvas.bind("<B1-Motion>", self.paint)

	def paint(self, event):
		x, y = (event.x), (event.y)
		r = 15
		self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

	def addBottons(self):
		clearButton = Button(self.master, text="Clear", command=self.clearCanvas, height=1, width=5)
		clearButton.place(x=0, y=0)

		exportImageButton = Button(self.master, text="Detect Number", command=self.getImage, height=1, width=10)
		exportImageButton.place(x=80, y=0)

	def clearCanvas(self):
		self.canvas.delete("all")

	def getImage(self):
		"""
			Convertes the canvascontent to a postscript and then to an image. 
		"""
		global root, image

		postscript = self.canvas.postscript(colormode="gray")
		image = Image.open(io.BytesIO(postscript.encode('utf-8')))

		root.destroy()

class Menu:

	def drawMenu():

		# subprocess.run( "clear")
		print("--------------")
		print("     Menu     ")
		print("--------------")
		print("1. Draw a numbe and feed it to the neural network.")
		print("2. Train neural network")
		print("9. Exit")
		print("--------------")

		i = int(input("Please select an option from above\n"))

		if i == 1:
			# Input menu

			if getInputImage():
				global image, neuralNetwork

				# neuralNetwork = NeuralNetwork()

				neuralNetwork.createNetwork()

				data = neuralNetwork.imageToMnist(image)

				with tf.Session() as sess:

					sess.run(tf.global_variables_initializer())

					result = neuralNetwork.feedToNetwork(neuralNetwork.x)

					foo = sess.run(neuralNetwork.output, feed_dict={neuralNetwork.x: [data]})

					print("out", foo)
					print("result", result)
					print("resrun", sess.run(result))

				Menu.drawMenu()

			else:
				# Did not get image
				Menu.drawMenu()

		elif i == 2:
			# Train cnn
			neuralNetwork.createNetwork()
			neuralNetwork.trainNetwork()

			Menu.drawMenu()

		elif i == 9:
			# Exit
			sys.exit()


class NeuralNetwork:

	def createNetwork(self):

		print("Creating neural network")

		# Nomber of nodes in a specific layer
		numNodesImp = 784

		numNodesHl1 = 500
		numNodesHl2 = 500
		numNodesHl3 = 500

		numNodesOut = 10

		# Import node
		self.x = tf.placeholder("float")

		self.hl1 = {"weights": tf.Variable(tf.random_normal([numNodesImp, numNodesHl1])),
				"biases": tf.Variable(tf.random_normal([numNodesHl1]))}

		# self.hl2 = {"weights": tf.Variable(tf.random_normal([numNodesHl1, numNodesHl2])),
		# 		"biases": tf.Variable(tf.random_normal([numNodesHl2]))}

		# self.hl3 = {"weights": tf.Variable(tf.random_normal([numNodesHl2, numNodesHl3])),
		# 		"biases": tf.Variable(tf.random_normal([numNodesHl3]))}

		self.out = {"weights": tf.Variable(tf.random_normal([numNodesHl1, numNodesOut])),
		"biases": tf.Variable(tf.random_normal([numNodesOut]))}

	def feedToNetwork(self, data):

		print("Evaluating input in cnn")

		l1 = tf.add(tf.matmul(data, self.hl1["weights"]), self.hl1["biases"])
		l1 = tf.nn.relu(l1)

		# l2 = tf.add(tf.matmul(l1, self.hl2["weights"]), self.hl2["biases"])
		# l2 = tf.nn.relu(l2)

		# l3 = tf.add(tf.matmul(l2, self.hl3["weights"]), self.hl3["biases"])
		# l3 = tf.nn.relu(l3)

		self.output = tf.add(tf.matmul(l1, self.out["weights"]), self.out["biases"])
		self.output = tf.nn.relu(self.output)

		return self.output

	def trainNetwork(self):

		print("Train neural network")

		from tensorflow.examples.tutorials.mnist import input_data

		mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

		y = tf.placeholder("float")
		batch_size = 100

		currentOut = neuralNetwork.feedToNetwork(neuralNetwork.x)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=currentOut, labels=y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)

		numEpochs = 10
		batchSize = 100

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())

			for epoch in range(numEpochs):
				epochLoss = 0
				for _ in range(int(mnist.train.num_examples / batchSize)):
					epochX, epochY = mnist.train.next_batch(batchSize)

					_, c = sess.run([optimizer, cost], feed_dict={neuralNetwork.x: epochX, y: epochY})
					print("cost", c)
					epochLoss += c

				print("Epoch", epoch, "completed out of", numEpochs, "loss:", epochLoss)

			correct = tf.equal(tf.argmax(currentOut, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, "float"))
			print("Accuracy:", accuracy.eval({neuralNetwork.x: mnist.test.images, y: mnist.test.labels}))

	def imageToMnist(self, image):
		"""
			Loads the image, downscales it to 28x28px, converts it to a numpy rgb array, converts it to a grayscale double flattened array
		"""
		side = (28, 28)
		image = image
		image.thumbnail(side, PIL.Image.ANTIALIAS)

		rbgImageArray = numpy.asarray(image)
		imageArray = []

		# TODO check if flattened correctly
		for row in rbgImageArray:
			for column in row:
				imageArray.append(float(column[0] / 255))

		return imageArray

def getInputImage():
	"""
		Opens the drawing board and returns a boolean depending on whether we got an image or not
	"""
	global root, image

	image = None # Set to nil so that we can later check whether we received an image or not

	root = Tk()

	root.geometry("500x540")
	root.resizable(width=False, height=False)

	app = Window(root)
	root.mainloop()

	if image:
		print("Got image")
		return True

	else:
		# print("Did not get an image")
		return False

if __name__ == '__main__':

	neuralNetwork = NeuralNetwork()

	Menu.drawMenu()
	