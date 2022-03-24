from PIL import Image
import numpy
from numpy import array #Added this line so array doesn't give an undefined error when used on line 11
import pandas

numpy.set_printoptions(threshold=numpy.inf)#Sets the length at which an array is truncated, while printing, to infinity

def format_Image(img):#Turned the file into a method
    simg = img.resize((64,64),resample=Image.NEAREST)#Resize the image to 64x64 pixels using the Nearest Function from PIL
    m = array(simg)#Define an array from the image
    simg.show()#Show the image

    simg.convert("P", palette=Image.ADAPTIVE, colors=8) # just temporary, later we can convert array values to hexidecimal to get a single channel 4 bit per pixel image
    simg.show