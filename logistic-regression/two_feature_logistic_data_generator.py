import csv
import numpy as np
import matplotlib.pyplot as plt

X1 = np.random.rand(100) * 20 - 10
X2 = np.random.rand(100) * 20 - 10

Y = np.vectorize(lambda x: 0 if x <= .5 else 1)(1 / (1 + np.exp(-(1 * X1 + X2 * 2))))

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, Y)
plt.show()


# transpose data
data = np.array([X1, X2, Y])
data_T = data.transpose()
# write file
with open('multiple_feature_lregression_data.csv', mode='x', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x1', 'x2', 'y'])
    writer.writerows(data_T)