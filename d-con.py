import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


mat = np.loadtxt('heart.ldt', skiprows=1)
matrix = np.array(mat)
x0 = np.array(mat)
x = x0[:,0]
a = 100
b = 0.5
y = a * x + b
model = svm.SVR(C=1.0, kernel='linear', epsilon=0.1)    # 正則化パラメータ=1, 線形カーネルを使用
model.fit(x.reshape(-1, 1), y)                          # フィッティング

# 学習済モデルを使って予測
x_reg = np.arange(0, 10, 1)                             # 回帰式のx軸を作成
y_reg = model.predict(x_reg.reshape(-1, 1))             # 予測
r2 = model.score(x.reshape(-1, 1), y)                   # 決定係数算出

# ここからグラフ描画---------------------------------------------------------------
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# グラフの上下左右に目盛線を付ける。
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')

# 軸のラベルを設定する。
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# データプロットの準備。
ax1.scatter(x, y, label='Dataset', lw=1, marker="o")
ax1.plot(x_reg, y_reg, label='Regression curve', color='red')
plt.legend()
plt.text(0.5, 7, '$\ R^{2}=$' + str(round(r2, 2)), fontsize=20)

# レイアウト設定
fig.tight_layout()

# グラフを表示する。
plt.show()
plt.close()
plt.savefig('popopo.png')
