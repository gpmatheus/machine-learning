import numpy as np
import pandas
import sys

data = pandas.read_csv(sys.argv[1] if len(sys.argv) == 2 else 'system.csv')

X = data.drop('y', axis=1).values
y = data['y'].values

# Step 1: Compute X^T * X
XT_X = np.dot(X.T, X)

# Step 2: Compute X^T * y
XT_y = np.dot(X.T, y)

# Step 3: Compute (X^T * X)^-1
XT_X_inv = np.linalg.inv(XT_X)

# Step 4: Compute the coefficients theta
theta = np.dot(XT_X_inv, XT_y)

print("Coefficients (theta):", theta)
