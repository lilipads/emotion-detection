import numpy as np
import Image
import pylab as pl
import csv

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
for example, vectorize('webcam-0.jpg')
"""
def vectorize(filename):
    size = 28, 10 # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    # pl.imshow(im_array, cmap=cm.Greys_r)
    # pl.show()
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array