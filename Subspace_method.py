# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:33:18 2021

@author: yuuki
"""

import numpy as np
from sm_eigen2 import eigen
#import sys
#CR(累積寄与率のパラメータ), data_tra(教師データラベル付き), k_arr_tst(テストデータ), y_tst(テストデータのクラスラベル),
#基底ベクトルの重みをどうするか(1ならば固有値を重み、それ以外なら重みを１とする)
def Subspace_method(CR,data_tra,k_arr_tst,y_tst,omomi_taku):
    #クラス数を数える
    class_num=int(np.max(y_tst))+1
    for i in range(class_num):
        #クラスiのデータを取り出す
        class_data_tra=data_tra[data_tra[:,-1]==i]
        class_x_tra=np.delete(class_data_tra,-1,1)
        
        #クラスiの固有値と固有ベクトルを求める
        eigen_values,eigen_vectors = eigen(class_x_tra)
        #print(eigen_vectors)
        #寄与率を取る
        sum_eigen_values=np.sum(eigen_values)
        #固有値の数
        eigen_value_num=len(eigen_values)
        #固有値の最大値
        max_eigen_value=eigen_values[0]
        kijun=0
        CR_per=CR
        for j in range(1,eigen_value_num+1):
            if CR_per==100:
                eigen_values=eigen_values[:]
                eigen_vectors=eigen_vectors[:,:]
                #j=eigen_value_num+1
                break
            kijun=eigen_values[j-1]+kijun
            kijun2=(kijun/sum_eigen_values)*100
            
            if kijun2>=CR_per:
                #j=j+1
                eigen_values=eigen_values[0:j]
                eigen_vectors=eigen_vectors[0:j,:]
                break
        
        #重みを固有値とする場合
        if omomi_taku==1:
            eigen_values=eigen_values/max_eigen_value
            #print(eigen_values)
            num_eig_val=len(eigen_values)
            #テストデータ数
            num_tst=k_arr_tst.shape[0]
            omomi_box=np.zeros([num_eig_val,num_tst])
            for k in range(num_eig_val):
                omomi_box[k,:]=eigen_values[k]
            projection_length=np.dot(eigen_vectors, k_arr_tst.T)
            #各クラスの射影長を求める
            projection_length=np.multiply(omomi_box,projection_length)
            
        if omomi_taku!=1:    
            #各クラスの射影長を求める
            #[本数, 次元数]、[次元数、データ数]
            projection_length=np.dot(eigen_vectors, k_arr_tst.T)
        projection_length=np.linalg.norm(projection_length,axis=0)
        projection_length=np.power(projection_length,2)
        if i==0:
            Projection_length_matrix=projection_length
        if i!=0:
            Projection_length_matrix=np.c_[Projection_length_matrix,projection_length]
    #print(Projection_length_matrix)
    #射影長の二乗和が大きい方のクラスに分類
    ninsikiresult=Projection_length_matrix.argmax(1)
    #print(ninsikiresult)
    #(テストデータのラベル)と(分類した配列)を比較する。
    Comparison=np.equal(y_tst,ninsikiresult)
    #print(Comparison)
    #比較して合致する個数.
    Agreement=len((np.where(Comparison == True))[0])
    #認識率を測定
    recognition_rate=Agreement/len(y_tst)
    
    return recognition_rate
    
        
        
        
        
        
        
        
        