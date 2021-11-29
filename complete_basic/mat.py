import numpy as np
import kernel
print('please input letter')
print('liner')
print('poly')
print('rbf')
method = input('')
if method in ['p', 'P', 'poly', 'POLY']:
    print('please input number of "d"')
    str = input('')
    d = int(str)
elif method in ['r', 'R', 'rbf', 'rfb']:
    print('please input number of "r"')
    str = input('')
    r = int(str)
else:
    breakpoint


arr = np.loadtxt("banana.txt",skiprows=1)
x = np.array([data[0] for data in arr])
y = np.array([data[1] for data in arr])
q = np.array([x, y])
qt = q.T
result = np.zeros((len(qt), len(qt)))
# result = np.array(q)

for i in range(len(qt)):
    for j in range(i,len(qt)):
        if method in ['l', 'L', 'liner', 'LINER']:
            result[i][j] = kernel.liner(qt[i],qt[j])
            result[j][i] = result[i][j]
        elif method in ['p', 'P', 'poly', 'POLY']:
            result[i][j] = kernel.poly(qt[i],qt[j], d)
            result[j][i] = result[i][j]
        elif method in ['r', 'R', 'rbf', 'rfb']:
            result[i][j] = kernel.ln(qt[i],qt[j], r)
            result[j][i] = result[i][j]           
print(result)
print(result.shape)