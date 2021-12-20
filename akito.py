import numpy as np
import numpy.linalg as la

#データ読み込みとクラス分け
data = np.loadtxt('heart.ldt', skiprows = 1)
rate = float(input('rate[%]-->')) / float(100)

data0 = np.array(data[data[:, -1] == 0, :-1])
data1 = np.array(data[data[:, -1] == 1, :-1])

#クラスごとに行列Aを出して固有値、固有ベクトルを求める
A0 = np.dot(data0.T, data0) / data0.shape[0]
A1 = np.dot(data1.T, data1) / data1.shape[0]

l0, p0 = la.eig(A0)
l1, p1 = la.eig(A1)

#固有値と固有ベクトルを降順ソート
L0 = l0[l0.argsort()[::-1]]
# P0 = p0[l0.argsort()[::-1]]

L1 = l1[l1.argsort()[::-1]]
# P1 = p1[l1.argsort()[::-1]]
index0 = np.argsort(l0)[::-1]
index1 = np.argsort(l1)[::-1]

#固有値の合計を求めて累積寄与率まで足した時のindexをi, jに保存
total0 = sum([i for i in L0 if i >= 0.0])
total1 = sum([i for i in L1 if i >= 0.0])

mul0 = 0.0
mul1 = 0.0

eig0 = []
eig1 = []

for i in range(L0.shape[0]):
    mul0 += L0[i]
    eig0.append(p0.T[index0[i]])
    if mul0 / total0 >= rate:
        break

for j in range(L1.shape[0]):
    mul1 += L1[j]
    eig1.append(p1.T[index1[i]])
    if mul1 / total1 >= rate:
        break

#i,jまでの固有ベクトルと全データとの内積
map0 = np.dot(eig0, data[:, :-1].T)
map1 = np.dot(eig1, data[:, :-1].T)

map0_leng = np.zeros(map0.shape[1])
map1_leng = np.zeros(map1.shape[1])

#射影長の自乗和をデータごとに求める
# for k0 in range(map0.shape[1]):
#     for m0 in range(map0.shape[0]):
#         map0_leng[k0] += pow(map0[m0,k0], 2)

# for k1 in range(map1.shape[1]):
#     for m1 in range(map1.shape[0]):
#         map1_leng[k1] += pow(map1[m1,k1], 2)

#射影長の自乗和を基にデータのラベリング
class_num = np.zeros(data.shape[0])
num = 170
for i in range(num):
        if la.norm(map0.T[i]) > la.norm(map1.T[i]):
            class_num[i] = 0
        else:
            class_num[i] = 1
num0 = 0
for i in range(num):
        if class_num[i] == data[i,-1]:
            num0 = num0 + 1
        else: pass
result = num0*100 / num
print(result)
# for n in range(data.shape[0]):
#     if map0_leng[n] >= map1_leng[n]:
#         class_num[n] = 0
#     else:
#         class_num[n] = 1

# #元データとの比較と認識率を求める
# rec_num = 0
# for p in range(data.shape[0]):
#     if class_num[p] - data[p, -1] == 0:
#         rec_num += 1

# rec_rate = rec_num * 100 / data.shape[0]   
# print(rec_rate)