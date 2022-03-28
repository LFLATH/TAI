'''
def lineYeild(string, f):
    for line in f:
        if line == string:
            yield line
        else:
            print('ERROR: a jpg RGB 8 bit file is needed')

f = open('C:/Users/Sweet/source/repos/TAI/TAI/Test_Images/t1.jpg','rb')
    
for line in lineYeild(b'\xff\xc0\x00\x11\x08', f):
    print(line)


'''

img = open('C:/Users/Sweet/source/repos/TAI/TAI/Test_Images/tomato1.jpg','rb')
img = img.read()
index = img.find(b'\xff\xc0\x00\x11\x08')

if img.find(b'\xff\xc0\x00\x11\x08') > -1:
    print('yay, RGB 8bit image')
else:
    print('ERROR: a jpg RGB 8 bit file is needed')

properties = img[index:index+85]



    



#print(' '.join(map(str,?fix?)))#turns into decimal