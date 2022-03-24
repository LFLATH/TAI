from PIL import Image
import numpy
import pandas

numpy.set_printoptions(threshold=numpy.inf)

img = Image.open(r"C:/users/Sweet/source/repos/assets/test1.jpg")

simg = img.resize((64,64),resample=Image.NEAREST)
m = array(simg)
simg.show()

simg.convert("P", palette=Image.ADAPTIVE, colors=8) # just temporary, later we can convert array values to hexidecimal to get a single channel 4 bit per pixel image

simg.show
