import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

data = np.loadtxt("/Users/huangguocheng/PycharmProjects/mathModel/latest/train_data.csv", delimiter=",")

train_label = np.array(data[:, -1])
train_label = train_label.astype(np.int).reshape(-1, 1)
train_data = np.array(data[:, 0:-1])

# 标准化
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)

# 创建神经网络，并训练
clf = RandomForestClassifier(max_depth=10)
clf.fit(train_data, train_label)

# 测试
test = np.loadtxt("/Users/huangguocheng/PycharmProjects/mathModel/temporary/test_data/S1/S1c1_test_data.csv",
                  delimiter=",")
test = np.delete(test, 0, 0)
test_row_col = np.array(test[:, -1])
test_data = np.array(test[:, :-1])

# 测试数据标准化
scaler = MinMaxScaler()
test_data = scaler.fit_transform(test_data)

predict = clf.predict(test_data)
print()
