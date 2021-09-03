import scipy.io
import numpy as np

mat = scipy.io.loadmat('label.mat')
matrix = mat['label']
np.save(open('label.npy', 'wb'), matrix)


# x = np.load(open('label.npy', 'rb'))
