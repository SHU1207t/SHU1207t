import matplotlib.pyplot as plt
import numpy as np

mat = np.loadtxt('heart.ldt', skiprows=1)
matrix = np.array(mat[:,:-1])
mat0 = np.array([data[:-1] for data in mat if data[-1] == 0 ])
mat1 = np.array([data[:-1] for data in mat if data[-1] == 1 ])
amat0 = np.dot(mat0.T,mat0)/mat0.shape[0]
amat1 = np.dot(mat1.T,mat1)/mat1.shape[0]
e0, v0 = np.linalg.eig(amat0)
e1, v1 = np.linalg.eig(amat1)
esum0 = np.sum(e0)
esum1 = np.sum(e1)
index = np.argsort(e0)[::-1]
index = np.argsort(e1)[::-1]
i, j = 0, 0
total0, total1 = 0, 0
while total0/esum0 < 0.95:
    if e0[index[i]] < 0:
        e0[index[i]] = 0
    total0 = total0 + e0[index[i]]
    i = i + 1
count0 = i - 1
while total1/esum1 < 0.95:
    if e0[index[i]] < 0:
        e0[index[i]] = 0
    total1 = total1 + e1[index[j]]
    j = j + 1
count1 = j - 1
total0, total1 = 0, 0
eig0 = np.array(v0[:count0])
eig1 = np.array(v1[:count1])
m0 = np.dot(matrix,eig0[:count0].T)
tmp0 = np.dot(m0,m0.T)
m1 = np.dot(matrix,eig1[:count1].T)
tmp1 = np.dot(m1,m1.T)
length0 = np.linalg.norm(tmp0)
length1 = np.linalg.norm(tmp1)
# print(length0)
# print(length1)
if length0 < length1:
    /mat0.shape[0]
else: