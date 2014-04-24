import net
import numpy as np

train_data = [(np.array([1,1,1,1,1]),np.array([0,0,0,0,1])), 
              (np.array([1,1,0,1,1]),np.array([0,0,0,1,0]))]

detector = net.Net([5,3,5])
detector.SGD(train_data, 2, 10, 0.1)
