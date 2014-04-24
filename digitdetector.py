# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import net
from array import array
import gzip
import struct
import numpy as np

# <codecell>

# open images
f = gzip.open('./train-images-idx3-ubyte.gz', 'rb') # test set
_, n_images, rows, cols = struct.unpack('>IIII', f.read(16)) # first 16 bytes are metadata
pixels = np.array(array('B', f.read())) / 255.0 # pixel data are unsigned bytes; normalize to [0,1] scale
f.close()
images = pixels.reshape((n_images, rows * cols)) # each image is a vector of 28*28 values in [0,1]

# <codecell>

f = gzip.open('./train-labels-idx1-ubyte.gz', 'rb')
_, n_images = struct.unpack('>II', f.read(8)) # first 16 bytes are metadata
digits = np.array(array('B', f.read())) 
f.close()

# <codecell>

labels = np.zeros((n_images, 10))
for i in range(n_images):
    labels[i][digits[i]] = 1

# <codecell>

temp_images = images[0:10]
temp_labels = labels[0:10]
train_data = zip(temp_images, temp_labels)

# <codecell>

detector = net.Net([rows * cols, 400, 10])
detector.SGD(train_data, 2, 10, 0.1)