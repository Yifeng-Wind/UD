import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_sum = sum(np.exp(L))
    return [np.exp(i)*1.0/exp_sum for i in L]
