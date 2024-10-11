from sklearn import linear_model
import pandas
import matplotlib.pyplot as plt
import numpy as np

data = pandas.read_csv('multiple_feature_lregression_data.csv')
X = data.drop('y', axis=1)
Y = data['y']

logr = linear_model.LogisticRegression()
logr.fit(X, Y)

theta = logr.coef_[0]

X1 = X['x1']
X2 = X['x2']

# plot dots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X['x1'], X['x2'], Y, color='b', marker='o', label='Data')

# plot surface

def f(theta, x, y):
    return 1 / (1 + np.exp(-(theta[0] * x + theta[1] * y)))

x_surface = np.linspace(np.min(X1), np.max(X1), 1000)
y_surface = np.linspace(np.min(X2), np.max(X2), 1000)
x_surface, y_surface = np.meshgrid(x_surface, y_surface)
z_surface = f(theta, x_surface, y_surface)

ax.plot_surface(x_surface, y_surface, z_surface, color='r', alpha=0.5)

plt.show()