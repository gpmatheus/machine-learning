import numpy as np

x = np.linspace(0, np.pi * 4, 1000)
y = np.sin(x)

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()

import csv

with open('sin.csv', mode='x', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])
    data = np.array([x, y]).transpose()
    for i in data:
        writer.writerow(i)