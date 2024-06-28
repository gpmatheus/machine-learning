import random
import csv
import numpy as np

X = [random.random() for i in range(50)]
data = np.array([X, [i * .4 + 3 for i in X]])
data = data.transpose()

with open('simple_data.csv', mode='x', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])
    for i in data:
        writer.writerow(i)