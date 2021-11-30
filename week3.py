from re import I
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg.linalg import eig

mat = np.loadtxt('heart.ldt', skiprows=1)
que = np.array(mat[:, :-1])
cmat = np.cov(que, rowvar=0, bias=0)
e, v = np.linalg.eig(cmat)
esum = np.array(np.sum(e))

index = np.array(np.sort(e)[::-1])
total = 0
i = 0
while total/esum <= 0.95:
    total = total + index[i]
    i = i + 1
i = i - 1
print(i)
eig = np.array(v[])