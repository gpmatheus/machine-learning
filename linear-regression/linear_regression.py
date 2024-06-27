import sys

args = {
    'seed': 1,
    'file': 2,
    'x': 3,
    'y': 4,
    'alpha': 5,
    'max_dvt': 6
}

if len(sys.argv) < len(args) + 1:
    print('Args required:')
    for key, value in args.items():
        print(f'{key}[{value}]')
    exit()

import pandas
companies = pandas.read_csv(sys.argv[args['file']])

X = companies[sys.argv[args['x']]]
Y = companies[sys.argv[args['y']]]

import numpy as np
data = np.array([X, Y])

def create_hypothesis(theta):
    def hipothesis(x):
        return theta[0] + theta[1] * x
    return hipothesis

def dvt(data, h):
    theta0_dvt = 0.0
    theta1_dvt = 0.0
    x = data[0]
    y = data[1]
    data_len = data.shape[1]
    for i in range(data_len):
        temp = h(x[i]) - y[i]
        theta0_dvt += temp
        theta1_dvt += temp * x[i]
    return (theta0_dvt / data_len), (theta1_dvt / data_len)

np.random.seed(int(sys.argv[args['seed']]))
theta0, theta1 = np.random.rand(), np.random.rand()
h = create_hypothesis([theta0, theta1])
theta0_dvt, theta1_dvt = dvt(data, h)
alpha = float(sys.argv[args['alpha']]) # 0.0001

max_dvt = float(sys.argv[args['max_dvt']]) # 0.0001
while theta0_dvt > max_dvt or theta1_dvt > max_dvt:
    theta0 -= theta0_dvt * alpha
    theta1 -= theta1_dvt * alpha
    h = create_hypothesis([theta0, theta1])
    theta0_dvt, theta1_dvt = dvt(data, h)

print(theta0, theta1)

new_X = np.linspace(0, max(X), 2)
new_Y = h(new_X)

import matplotlib.pyplot as plt
plt.title('Linear Regression Model')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.scatter(X, Y)
plt.plot(new_X, new_Y, color='red')
plt.show()