import pandas
import numpy as np
import argparse
import sys

# Linear Regression class implementation
class LinearRegression:

    def __init__(self, data, seed=None, x0name='x0', target='y', alpha=0.01, max_dvt=0.01):
        self.__X = data.drop(target, axis=1)
        self.__Y = data[target]
        self.__data_len = len(self.__Y)
        self.__X.insert(0, x0name, np.ones(self.__data_len))
        self.__alpha = alpha
        self.__max_dvt = max_dvt
        np.random.seed(seed)
        self.__theta = np.random.rand(self.__X.shape[1])

    def __create_hypothesis(self, theta):
        def hypothesis(X):
            return np.dot(theta, X)
        return hypothesis

    def __dvt(self, h):
        def derivative(X, y, j):
            return (h(X) - y) * X[j]
        return np.array(
            [np.sum([derivative(x, y, j) for x, y in zip(self.__X.values, self.__Y)]) / self.__data_len for j in range(self.__X.shape[1])])
    
    def train(self):
        h = self.__create_hypothesis(self.__theta)
        derivatives = self.__dvt(h)
        while np.any(np.abs(derivatives) > self.__max_dvt):
            derivatives *= self.__alpha
            self.__theta -= derivatives
            h = self.__create_hypothesis(self.__theta)
            derivatives = self.__dvt(h)
        self.__derivatives = derivatives
    
    def result(self):
        return self.__theta, self.__derivatives
    
    def guess(self, x):
        if len(x) == len(self.__theta) - 1:
            theta = list(self.__theta)
            theta0 = theta.pop(0)
            return theta0 + np.dot(np.array(theta), x)
        return None


# test

def plot3d(x, y, z, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, color='b', marker='o', label='Data')

    x_surface = np.linspace(np.min(x1), np.max(x1), 2)
    y_surface = np.linspace(np.min(x2), np.max(x2), 2)
    x_surface, y_surface = np.meshgrid(x_surface, y_surface)
    z_surface = theta[0] + x_surface * theta[1] + y_surface * theta[2]

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('data and resulting surface')
    ax.set_zlim(np.min(z_surface) - 1, np.max(z_surface) + 1)
    ax.plot_surface(x_surface, y_surface, z_surface, color='r', alpha=0.5)

    plt.show()


def plot2d(x, y, theta):
    new_x = np.linspace(np.min(x), np.max(x), 2)
    new_y = theta[0] + theta[1] * new_x
    plt.set_xlabel('x')
    plt.set_ylabel('y')
    plt.set_title('data and resulting line')
    plt.scatter(x, y)
    plt.plot(new_x, new_y, color='red')
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog="Multiple Feature Linear Regression",
        description="Trains a Linear Model to Match Data")
    parser.add_argument('--filename', action='store', nargs='?', required=True, type=str)
    parser.add_argument('--x0name', action='store', nargs='?', required=False, default='x0', type=str)
    parser.add_argument('--target', action='store', nargs='?', required=True, type=str)
    parser.add_argument('--seed', action='store', nargs='?', required=False, type=int)
    parser.add_argument('--alpha', action='store', nargs='?', required=False, default=0.01, type=float)
    parser.add_argument('--max-dvt', action='store', nargs='?', required=False, default=0.01, type=float)
    parser.add_argument('--plot', action='store', nargs='?', required=False, default=True, type=bool)
    parser.add_argument('--sep', action='store', nargs='?', required=False, default=',', type=str)

    sys.argv.pop(0)
    args = parser.parse_args(sys.argv)
    data = pandas.read_csv(args.filename, sep=args.sep)

    linear_regression = LinearRegression(
        data,
        seed=args.seed, 
        x0name=args.x0name, 
        target=args.target, 
        alpha=args.alpha, 
        max_dvt=args.max_dvt)

    linear_regression.train()
    theta, derivatives = linear_regression.result()

    print('Resulting theta:')
    print(['{:.20f}'.format(t) for t in theta], '\n')

    if args.plot:
        import matplotlib.pyplot as plt
        if len(theta) > 3:
            print('Data can\'t be plotted. Too many dimensions.')
        elif len(theta) == 3:
            x1 = data.iloc[:, 0]
            x2 = data.iloc[:, 1]
            y = data[args.target]
            plot3d(x1, x2, y, theta)
        elif len(theta) == 2:
            x = data.iloc[:, 0]
            y = data[args.target]
            plot2d(x, y, theta)