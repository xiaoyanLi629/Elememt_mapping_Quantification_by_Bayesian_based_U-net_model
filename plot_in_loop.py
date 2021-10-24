import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

x = []
y = []

for i in range(20):
	x.append(i)
	y.append(np.random.random())
	print(i)
	# animate(x, y)
	ax1.clear()
	plt.scatter(x, y)
	plt.pause(0.001)
plt.show()
