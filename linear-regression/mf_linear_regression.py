import pandas
import numpy as np

data = pandas.read_csv('multiple_feature_data.csv')
data.insert(0, 'x0', np.ones(data.shape[0]))

def create_hypothesis(theta):
    def hypothesis(X):
        return np.dot(theta, X)
    return hypothesis

def dvt(data, h):
    def derivative(X, y, j):
        return (h(X) - y) * X[j]
    X = data.drop('y', axis=1)
    Y = data['y']
    data_len = X.shape[0]
    return np.array([np.sum([derivative(x, y, j) for x, y in zip(X.values, Y)]) / data_len for j in range(X.shape[1])])

# np.random.seed(0)
theta = np.random.rand(data.shape[1] - 1)
h = create_hypothesis(theta)
derivatives = dvt(data, h)

alpha = 0.001
max_dvt = 0.0001
while np.all(np.abs(derivatives) > max_dvt):
    derivatives *= alpha
    theta -= derivatives
    h = create_hypothesis(theta)
    derivatives = dvt(data, h)

# print trained data
print(theta)

# plot result
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = data.drop('x0', axis=1)
x1 = data['x1']
x2 = data['x2']
y = data['y']
ax.scatter(x1, x2, y, color='b', marker='o', label='Data')

x_surface = np.linspace(min(x1), max(x1), 2)
y_surface = np.linspace(min(x2), max(x2), 2)
x_surface, y_surface = np.meshgrid(x_surface, y_surface)
z_surface = theta[0] + x_surface * theta[1] + y_surface * theta[2]
ax.plot_surface(x_surface, y_surface, z_surface, color='r', alpha=0.5)

plt.show()