import os
import re
from more_itertools import sort_together
import numpy as np

lines = np.asarray(np.loadtxt("checkpoints/scores.txt", comments="#", delimiter="-", unpack=False)).astype(int)
print(lines)
lines = lines[:,0]
print(lines)

for i, num in enumerate(lines):
	if i == 0:
		continue
	index = np.argmax(lines[0:i])
	print(index)
	print(lines[index])