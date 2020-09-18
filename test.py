import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

data = pd.read_excel("/Users/huangguocheng/PycharmProjects/mathModel/data/persons/S1/S1_train_data.xlsx", sheet_name=0,
                     header=None)
data=np.array(data)

# np.savetxt("collect.csv", data, delimiter=",", fmt='%f')

# 数据集
train_label = np.array(data[:, -1])
train_label = train_label.astype(np.int).reshape(-1, 1)
train_data = np.array(data[:, 0:-1])

pca = PCA(n_components=1)
pca = pca.fit(train_data)
Y_dr = pca.transform(train_data)

np.savetxt("test.csv", Y_dr, delimiter=",")

print()
