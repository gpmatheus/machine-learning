import csv
import numpy as np

X1 = np.random.rand(100) * 10
X2 = np.random.rand(100) * 10

Y = 2 + X1 * 1 + X2 * 3

data = np.array([X1, X2, Y])
data_T = data.transpose()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, Y, color='g')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('X1, X2 e Y')
plt.show()

with open('multiple_feature_data.csv', mode='x', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x1', 'x2', 'y'])
    for i in data_T:
        writer.writerow(i)