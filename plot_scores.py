import numpy as np
import matplotlib.pyplot as plt


lines = np.asarray(np.loadtxt("checkpoints/scores.txt", comments="#", delimiter="-", unpack=False)).astype(int)
print(lines)
lines = lines[:,0]
#lines = text_file.read().split(',')
#print(lines.shape)

plt.plot(lines, color='red')
plt.show()