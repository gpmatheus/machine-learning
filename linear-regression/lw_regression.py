# # import pandas
# # import numpy as np

# # data = pandas.read_csv('sin.csv')

# # x0 = np.ones(data.shape[0])
# # data.insert(0, 'x0', x0)
# # x = data.drop('y', axis=1).values
# # y = data['y'].values

# # def h(theta, x):
# #     return np.dot(theta, x)

# # theta = np.random.random(2)

# # weight = 0.0001
# # def w(data_x, x):
# #     return np.exp(-((data_x - x) ** 2) / (2 * weight))


# # def dvt(theta, x, data_x, data_y):
# #     wght = w(data_x, x)
# #     def derivative(X, y, j):
# #         return wght * (y - h(theta, X)) * X[j]
# #     data_len = 2
# #     return np.array([np.sum([derivative(xd, yd, j) for xd, yd in zip(data_x, data_y)]) / data_len for j in range(data_len)])

# # x_input = float(input('insert the x value: '))

# # alpha = 0.01
# # for i in range(1000):
# #     derivatives = dvt(theta, x_input, x, y)
# #     derivatives *= alpha
# #     theta -= derivatives

# # print(theta)

# # import matplotlib.pyplot as plt

# # x = data['x'].values
# # plt.plot(x, y)
# # # plt.scatter([x_input], [guess])
# # line_x = np.linspace(0, np.pi * 4, 1000)
# # line_y = line_x * theta[1]
# # line_y += theta[0]
# # plt.plot(line_x, line_y)
# # plt.show()

# import pandas
# import numpy as np

# data = pandas.read_csv('sin.csv', sep=',')

# X = data.drop('y', axis=1)
# X.insert(0, 'x0', np.ones(X.shape[0]))
# Y = data['y'].values

# # w * (y[i] - h(X)) * X[i]

# def weight_func(X, x_position, weight):
#     return np.exp(-((X - x_position) ** 2) / (2 * weight))

# weight = 0.0001
# def derivative(X, x_position, theta):
#     def feature_iterator(feature):
#         def data_iterator(row, y, feature):
#             w = weight_func(row, x_position, weight)
#             return w * (y - np.dot(theta, row)) * feature
#         values = X.iloc[:, feature]
#         target = Y
#         vectorized_data_iterator = np.vectorize(data_iterator)
#         vectorized_data_iterator(values, target, feature)
#     features = np.array(list(range(len(theta))))
#     return feature_iterator(features)

# x_position = float(input('type the x position: '))
# theta = np.random.random(X.shape[1])
# der = derivative(X, x_position, theta)

# print(der)

import pandas
import numpy as np

data = pandas.read_csv('sin.csv', sep=',')

X = data.drop('y', axis=1)
X.insert(0, 'x0', np.ones(X.shape[0]))
m = X.shape[0]
n = X.shape[1]
Y = data['y'].values

def weight_func(x, x_position, w):
    return np.exp(-((x - x_position) ** 2) / (2 * w))

weight = 0.001
x_positions = [2.0]
theta = np.random.rand(n)

for j in range(n):
    for i in range(m):
        w = weight_func(X[i][j], x_positions[j], weight)
        