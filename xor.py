import modelnet
import samplenet
from array import array
import gzip
import struct
import numpy as np

# create a network with two input, two hidden, and one output node
# sample code
pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]
n = samplenet.NN(2, 2, 1)
n.train(pat)
n.test(pat)

# our code

# pat = [
#         (np.array([0,0]), 0),
#         (np.array([0,1]), 1),
#         (np.array([1,0]), 1),
#         (np.array([1,1]), 0)
#     ]
# n = modelnet.Network([2, 2, 1])
# n.SGD(pat, 10000, 2, 1)
# print n.feedforward([0,1])
# print n.feedforward([1,0])
# print n.feedforward([0,0])
# print n.feedforward([1,1])


