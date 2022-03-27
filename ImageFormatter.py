from PIL import Image
import numpy
from numpy import array #Added this line so array doesn't give an undefined error when used on line 11
import pandas

numpy.set_printoptions(threshold=numpy.inf)#Sets the length at which an array is truncated, while printing, to infinity

def format_Image(img):#Turned the file into a method
    simg = img.resize((28,28),resample=Image.NEAREST)#Resize the image to 28x28 pixels using the Nearest Function from PIL
    return simg
    