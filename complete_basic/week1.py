fin=open('banana.ldt')
line=fin.read()
fio=open('banana.txt', 'w')
fio.write(line)
fin.close()

import numpy as np
import matplotlib.pyplot as plt



arr = np.loadtxt("banana.txt",skiprows=1)
x0 = np.array([data[0] for data in arr if data[-1] == 0 ])
y0 = np.array([data[1] for data in arr if data[-1] == 0 ])
x1 = np.array([data[0] for data in arr if data[-1] == 1 ])
y1 = np.array([data[1] for data in arr if data[-1] == 1 ])



#fig = plt.figure()

# ax = fig.add_subplot(1,1,1)

# ax.scatter(x,y)

# ax.set_title('first scatter plot')
# ax.set_xlabel('x')
# ax.set_ylabel('y')

# fig.savefig('plot.png')

plt.scatter(x0,y0, c = 'r')
plt.scatter(x1,y1, c = 'b')
plt.savefig('popo.png')