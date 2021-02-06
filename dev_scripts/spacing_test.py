import numpy as np

x = []
y = []
f = []
W = 10
N = W**2
f = 5

for i in range(1000):
    x.append(i%W)
    y.append(int(i/W)%W)
    f.append(int(i/N))

