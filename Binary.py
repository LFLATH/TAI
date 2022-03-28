'''
def lineYeild(string, f):
    for line in f:
        if line == string:
            yield line
        else:
            print('ERROR: a jpg RGB 8 bit file is needed.')

with open('C:/Users/Sweet/source/repos/TAI/TAI/Test_Images/t1.jpg','rb') as f:
    for line in lineYeild(b'\xff\xc0\x00\x11\x08', f):
        print(line)
'''
img = open('C:/Users/Sweet/source/repos/TAI/TAI/Test_Images/t1.jpg','rb')
img = img.read()
img.find(b'\xff\xc0\x00\x11\x08')



inum = (b'\xff\xc0\x00\x11\x08')





#print(' '.join(map(str,?fix?)))#turns into decimal