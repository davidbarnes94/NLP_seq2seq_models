import numpy as np

def sigmoid (x):
    return 1./(1 + np.exp(-x))

def softmax (x):
    return np.exp(x)/np.sum(np.exp(x))

x = np.array([0, 1.5, -2, 3.6, -9.55])

print(sigmoid(x))
print(softmax(x))
print('the sum of the probabilities for sigmoid is {0}'.format(np.sum(sigmoid(x))))
print('the sum of the probabilities for softmax is {0}'.format(np.sum(softmax(x))))