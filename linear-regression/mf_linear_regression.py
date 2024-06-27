import pandas
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(
    prog="Multiple Feature Linear Regression",
    description="Trains a Linear Model to Match Data")
parser.add_argument('--filename', action='store', nargs='?', required=True, type=str)
parser.add_argument('--x0name', action='store', nargs='?', required=False, default='x0', type=str)
parser.add_argument('--target', action='store', nargs='?', required=True, type=str)
parser.add_argument('--seed', action='store', nargs='?', required=False, default=1, type=int)
parser.add_argument('--alpha', action='store', nargs='?', required=False, default=0.01, type=float)
parser.add_argument('--max-dvt', action='store', nargs='?', required=False, default=0.01, type=float)

sys.argv.pop(0)
args = parser.parse_args(sys.argv)

data = pandas.read_csv(args.filename)
data.insert(0, args.x0name, np.ones(data.shape[0]))

def create_hypothesis(theta):
    def hypothesis(X):
        return np.dot(theta, X)
    return hypothesis

def dvt(data, h):
    def derivative(X, y, j):
        return (h(X) - y) * X[j]
    X = data.drop(args.target, axis=1)
    Y = data[args.target]
    data_len = X.shape[0]
    return np.array([np.sum([derivative(x, y, j) for x, y in zip(X.values, Y)]) / data_len for j in range(X.shape[1])])

np.random.seed(args.seed)
theta = np.random.rand(data.shape[1] - 1)
h = create_hypothesis(theta)
derivatives = dvt(data, h)

alpha = args.alpha
max_dvt = args.max_dvt
while np.all(np.abs(derivatives) > max_dvt):
    derivatives *= alpha
    theta -= derivatives
    h = create_hypothesis(theta)
    derivatives = dvt(data, h)

# print trained data
print(theta)

# plot result
data = data.drop(args.x0name, axis=1)
x_data = data.drop(args.target, axis=1)
if len(x_data.axes[1]) == 2:

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = x_data.iloc[:, 0]
    x2 = x_data.iloc[:, 1]
    y = data[args.target]
    ax.scatter(x1, x2, y, color='b', marker='o', label='Data')

    x_surface = np.linspace(min(x1), max(x1), 2)
    y_surface = np.linspace(min(x2), max(x2), 2)
    x_surface, y_surface = np.meshgrid(x_surface, y_surface)
    z_surface = theta[0] + x_surface * theta[1] + y_surface * theta[2]
    ax.plot_surface(x_surface, y_surface, z_surface, color='r', alpha=0.5)

    plt.show()