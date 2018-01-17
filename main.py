#! /usr/bin/env python3

from tkinter import *		# GUI
import io 					# Get Canvas Content
from PIL import Image
import PIL

import tensorflow as tf 	# Neural network

import sys					# Exiting
import subprocess			# Clear Terminal

import os
import os.path

import operator


from neuralNetwork import NeuralNetwork
import settings

root = None
image = None

neuralNetwork = None

settings.init()
saveFilePath = settings.saveFilePath

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
			Convertes the canvascontent to a postscript and then to an image. Called when btn pressed
		"""
		global root, image

		postscript = self.canvas.postscript(colormode="gray")
		image = Image.open(io.BytesIO(postscript.encode('utf-8')))


		root.destroy()

class Menu:

	def drawMenu():
		global image, neuralNetwork, saveFilePath

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

				data = neuralNetwork.imageToMnist(image)

				neuralNetwork.saver = tf.train.Saver({"hl1W": neuralNetwork.hl1["weights"], "hl1B": neuralNetwork.hl1["biases"], "hl2W": neuralNetwork.hl2["weights"], "hl2B": neuralNetwork.hl2["biases"], "hl3W": neuralNetwork.hl3["weights"], "hl3B": neuralNetwork.hl3["biases"], "outw": neuralNetwork.out["weights"], "outb": neuralNetwork.out["biases"]})

				with tf.Session() as sess:

					if (os.path.exists(saveFilePath) == False):
						print("There is no network model at %s. Initializing new one" % saveFilePath)
						sess.run(tf.global_variables_initializer())

					else:
						print("There is a network, will load it")
						neuralNetwork.saver.restore(sess, saveFilePath)
						print("missin", sess.run(tf.report_uninitialized_variables()))

					results = sess.run(neuralNetwork.feedToNetwork([data]))[0]

					index, value = max(enumerate(results), key=operator.itemgetter(1))

					# print("results", results)
					print("It seems to be a", index)

				Menu.drawMenu()

			else:
				# Did not get image
				Menu.drawMenu()

		elif i == 2:
			# Train cnn

			neuralNetwork.trainNetwork()

			Menu.drawMenu()

		elif i == 9:
			# Exit
			sys.exit()

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
		return False

if __name__ == '__main__':

	neuralNetwork = NeuralNetwork()

	Menu.drawMenu()
	