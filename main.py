#! /usr/bin/env python3

from tkinter import *
import io
from PIL import Image

root = None

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
		r = 6
		self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

	def addBottons(self):
		clearButton = Button(self.master, text="Clear", command=self.clearCanvas, height=1, width=5)
		clearButton.place(x=0, y=0)

		exportImageButton = Button(self.master, text="Detect Number", command=self.getImage, height=1, width=10)
		exportImageButton.place(x=80, y=0)

	def clearCanvas(self):
		self.canvas.delete("all")

	def getImage(self):
		global root

		postscript = self.canvas.postscript(colormode="gray")
		image = Image.open(io.BytesIO(postscript.encode('utf-8')))

		image.show()

		root.destroy()


	
def getInputImage():
	global root

	root = Tk()

	root.geometry("500x540")
	root.resizable(width=False, height=False)

	app = Window(root)
	root.mainloop()

	print("Called")


if __name__ == '__main__':

	getInputImage()
	