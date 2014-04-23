import net
from array import array
import gzip
import struct
import numpy as np

f = gzip.open('./train-images-idx3-ubyte.gz', 'rb') # test set
_, n_images, rows, cols = struct.unpack('>IIII', f.read(16)) # first 16 bytes are metadata
pixels = np.array(array('B', f.read())) / 255.0 # pixel data are unsigned bytes; normalize to [0,1] scale
f.close()

images = pixels.reshape((n_images, rows * cols)) # each image is a vector of 28*28 values in [0,1]
labels = gzip.open('./train-labels-idx1-ubyte.gz', 'rb') 

train_data = zip(images,labels)

detector = net.Net([784,500,2])
detector.SGD(train_data, 2, 10, 0.1)