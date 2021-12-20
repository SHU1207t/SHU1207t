import matplotlib.pyplot as plt
import numpy as np

# print('寄与率を入力せよ[%]')
# palameter = input('')
# palameter = int(palameter)/100
mat = np.loadtxt('heart.ldt', skiprows=1)
testdate = np.loadtxt('heart.tst', skiprows=1)
matrix = np.array(mat[:,:-1])
first  = int(1*matrix.shape[0]/5)
second = int(2*matrix.shape[0]/5)
third  = int(3*matrix.shape[0]/5)
forth  = int(4*matrix.shape[0]/5)
fifth  = int(5*matrix.shape[0]/5)
matrix1 = mat[:first]
matrix2 = mat[first:second]
matrix3 = mat[second:third]
matrix4 = mat[third:forth]
matrix5 = mat[forth:fifth]
def subspace(param, testmat, mat1, mat2, mat3, mat4):
    totalmat = np.r_[mat1,mat2,mat3,mat4]
    testmatrix = np.array(testmat[:,:-1])
    totalmat0 = np.array([data[:-1] for data in totalmat if data[-1] == 0 ])
    totalmat1 = np.array([data[:-1] for data in totalmat if data[-1] == 1 ])
    amat0 = np.dot(totalmat0.T,totalmat0)/totalmat0.shape[0]
    amat1 = np.dot(totalmat1.T,totalmat1)/totalmat1.shape[0]
    e0, v0 = np.linalg.eig(amat0)
    e1, v1 = np.linalg.eig(amat1)
    e0[e0<0] = 0
    e1[e1<0] = 0
    esum0 = np.sum(e0)
    esum1 = np.sum(e1)
    index0 = np.argsort(e0)[::-1]
    index1 = np.argsort(e1)[::-1]
    i, j = 0, 0
    total0, total1 = 0, 0
    eig0 = []
    eig1 = []
    while total0/esum0 <= param:
        total0 = total0 + e0[index0[i]]
        eig0.append(v0.T[index0[i]])
        i = i + 1
    while total1/esum1 <= param:
        total1 = total1 + e1[index1[j]]
        eig1.append(v1.T[index1[j]])
        j = j + 1
    
    total0, total1 = 0, 0
    num = testmatrix.shape[0]
    m0 = np.dot(testmatrix,np.array(eig0).T)
    m1 = np.dot(testmatrix,np.array(eig1).T)
    classdata = np.zeros(testmatrix.shape[0])
    for i in range(num):
        if np.linalg.norm(m0[i]) > np.linalg.norm(m1[i]):
            classdata[i] = 0
        else:
            classdata[i] = 1
    num0 = 0
    for i in range(num):
        if classdata[i] == mat[i,-1]:
            num0 = num0 + 1
    else: pass
    result = num0*100 / num
    return result
def eval_subspace(param, testmat, tmat):
    # totalmat = np.r_[mat1,mat2,mat3,mat4]
    testmatrix = np.array(testmat[:,:-1])
    totalmat0 = np.array([data[:-1] for data in tmat if data[-1] == 0 ])
    totalmat1 = np.array([data[:-1] for data in tmat if data[-1] == 1 ])
    amat0 = np.dot(totalmat0.T,totalmat0)/totalmat0.shape[0]
    amat1 = np.dot(totalmat1.T,totalmat1)/totalmat1.shape[0]
    e0, v0 = np.linalg.eig(amat0)
    e1, v1 = np.linalg.eig(amat1)
    e0[e0<0] = 0
    e1[e1<0] = 0
    esum0 = np.sum(e0)
    esum1 = np.sum(e1)
    index0 = np.argsort(e0)[::-1]
    index1 = np.argsort(e1)[::-1]
    i, j = 0, 0
    total0, total1 = 0, 0
    eig0 = []
    eig1 = []
    while total0/esum0 <= param:
        total0 = total0 + e0[index0[i]]
        eig0.append(v0.T[index0[i]])
        i = i + 1
    while total1/esum1 <= param:
        total1 = total1 + e1[index1[j]]
        eig1.append(v1.T[index1[j]])
        j = j + 1
    
    total0, total1 = 0, 0
    num = testmatrix.shape[0]
    m0 = np.dot(testmatrix,np.array(eig0).T)
    m1 = np.dot(testmatrix,np.array(eig1).T)
    classdata = np.zeros(testmatrix.shape[0])
    for i in range(num):
        if np.linalg.norm(m0[i]) > np.linalg.norm(m1[i]):
            classdata[i] = 0
        else:
            classdata[i] = 1
    num0 = 0
    for i in range(num):
        if classdata[i] == mat[i,-1]:
            num0 = num0 + 1
    else: pass
    result = num0*100 / num
    return result
# test = matrix1
# a = matrix2
# b = matrix3
# c = matrix4
# d = matrix5    
# reparam1 = subspace(palameter, test, a, b, c, d)

# test = matrix2
# a = matrix1
# b = matrix3
# c = matrix4
# d = matrix5 
# reparam2 = subspace(palameter, test, a, b, c, d)

# test = matrix3
# a = matrix1
# b = matrix2
# c = matrix4
# d = matrix5 
# reparam3 = subspace(palameter, test, a, b, c, d)

# test = matrix4
# a = matrix1
# b = matrix2
# c = matrix3
# d = matrix5 
# reparam4 = subspace(palameter, test, a, b, c, d)

# test = matrix5
# a = matrix1
# b = matrix2
# c = matrix3
# d = matrix4
# reparam5 = subspace(palameter, test, a, b, c, d)

# reparam = (reparam5 + reparam4 + reparam3 + reparam2 + reparam1)/5 
testreparam = eval_subspace(0.8, testdate, mat)
print('{:.1f}%'.format(testreparam))
# print('{:.1f}%'.format(reparam))