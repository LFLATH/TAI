from ImageFormatter import *
from numpy import asarray
from matplotlib import pyplot as plt
def makeArray(img): #We can plug in an image and convert that to a huearray
    img = format_Image(img)#img = format_Image(img)#Formats the Image
    data = asarray(img)#Makes image into an array of rgb values
    #print(data[0][0])#Prints the data
    #img.show()
    huearray = numpy.zeros((len(data) * len(data[0])))
    step = 0
    for i in range(len(data)):
        for j in range(len(data[0])):
            
            red = data[i][j][0]
            green = data[i][j][1]
            blue = data[i][j][2]
            red = red / 255
            green = green / 255
            blue = blue / 255
            Cmax = max(red, green, blue)
            Cmin = min(red, green, blue)
            delta = Cmax - Cmin
            if Cmax == Cmin:
                h = 0
            elif Cmax == red:
                h = (60*(((green - blue)/delta)%6))
            elif Cmax == green:
                h = (60*(((blue -red)/delta)+2))
            elif Cmax == blue:
                h = (60*(((red-green)/delta)+4))
            elif Cmax == 0:
                s = 0
            else:
                s = (delta / Cmax)
            v = Cmax
            huearray[step] = h
            step = step + 1
    return huearray