import numpy as np
def liner(x, y):
    scalar = np.dot(x, y)
    return scalar

def poly(x,y,d):
    scalar = np.dot(x, y)
    pol = (scalar + 1)**d
    return pol

def ln(x, y, r):

    ele1 = np.array(x) - np.array(y)
    ele2 = (np.linalg.norm(ele1))**2
    ele3 = -r*ele2
    result = np.exp(ele3)
    return result