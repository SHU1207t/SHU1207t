import matplotlib.pyplot as plt
import numpy as np

mat = np.loadtxt('heart.ldt', skiprows=1)
que = np.array(mat[:, :-1])
cmat = np.cov(que, rowvar=0, bias=0)
e, v = np.linalg.eig(cmat)
es = np.sort(e)[::-1]
index = np.argsort(e)[::-1]
mat0 = np.array([data[:-1] for data in mat if data[-1] == 0 ])
mat1 = np.array([data[:-1] for data in mat if data[-1] == 1 ])

vx0 = np.dot(v[index[0]],mat0.T)
vy0 = np.dot(v[index[1]],mat0.T)
vx1 = np.dot(v[index[0]],mat1.T)
vy1 = np.dot(v[index[1]],mat1.T)
plt.scatter(vx0,vy0,c='b')
plt.scatter(vx1,vy1,c='r')
plt.savefig('plot2.png')