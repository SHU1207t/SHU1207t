#部分空間法の固有値,固有ベクトルの関数

import numpy as np
#from scipy import linalg
import sys

def eigen(x):
    
    
    #Xに各データを代入
    #X_tXにXとX.Tの内積を代入
    X_tX=np.dot(x.T,x)
    
    #tani=np.eye(zigen)*0.000000001
    #X_tX=X_tX
    #X_tXをデータ数で割る
    X_tX=(X_tX)/x.shape[0]
    #X_tXの固有値問題を解く。
    #eigen_valuesに固有値、eigen_vectorsに固有ベクトルを代入
    eigen_values,eigen_vectors=np.linalg.eigh(X_tX)
    #1列目の１本目, 2列目が2本目
    eigen_vectors=eigen_vectors.T
    #print(eigen_vectors)
    #for eigen_num in range(len(eigen_values)):
        #eigen_vectors[eigen_num,:]=eigen_vectors[eigen_num]/np.linalg.norm(eigen_vectors[eigen_num])
    #固有値を大きい順に並べる。
    #各固有値に対応する固有ベクトルに並び替える。
    eigen_id = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[eigen_id]
    
    #固有ベクトルを並べ替える
    eigen_vectors = eigen_vectors[eigen_id,:]
    
    #固有値eigen_valuesの値が負ならば、0にする。
    eigen_values[eigen_values<0]=0
    
    #固有値が０の基底ベクトルを削除する
    eigen_0=np.where(eigen_values[:]==0)[0]
    
    #print(eigen_values)
    eigen_values=np.delete(eigen_values,eigen_0,0)
    
    eigen_vectors=np.delete(eigen_vectors,eigen_0,0)
    
    return eigen_values,eigen_vectors