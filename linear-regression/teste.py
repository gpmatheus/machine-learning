import numpy as np
import pandas as pd

# Definindo o tamanho do dataset
n = 100

# Gerando valores aleatórios para X1 e X2
np.random.seed(0)
X1 = np.random.rand(n) * 10  # valores de 0 a 10
X2 = np.random.rand(n) * 10  # valores de 0 a 10

# Calculando y como uma combinação linear de X1 e X2
y = 2 * X1 + 3 * X2

# Criando o DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
data.head()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plotando X1 vs y
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(data['X1'], data['y'], color='b')
plt.xlabel('X1')
plt.ylabel('y')
plt.title('X1 vs y')

# Plotando X2 vs y
plt.subplot(1, 2, 2)
plt.scatter(data['X2'], data['y'], color='r')
plt.xlabel('X2')
plt.ylabel('y')
plt.title('X2 vs y')

plt.show()

# Plotando X1, X2 e y em um gráfico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['X1'], data['X2'], data['y'], color='g')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('X1, X2 e y')

plt.show()