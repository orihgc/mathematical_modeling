import numpy as np
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import random

data = np.loadtxt("/Users/huangguocheng/PycharmProjects/mathModel/latest/S1train_data.csv", delimiter=",")

data_shape = data.shape

collect = np.zeros((1, 21))


for i in range(0, data_shape[0], 12):
    canReduce = 2
    for j in range(0, 12):
        if int(data[i + j, -1]) == 1:
            collect = np.vstack((collect, np.array(data[i + j])))
        elif canReduce > 0:
            collect = np.vstack((collect, np.array(data[i + j])))
            canReduce -= 1

data = np.delete(collect, 0, 0)

cov = np.cov(data)


# np.savetxt("collect.csv", data, delimiter=",", fmt='%f')

# 数据集
train_label = np.array(data[:, -1])
train_label = train_label.astype(np.int).reshape(-1, 1)
train_data = np.array(data[:, 0:-1])

# scaler = StandardScaler()
# train_data = scaler.fit_transform(train_data)

shape = train_data.shape

train_x, test_x, train_y, test_y = ms.train_test_split(train_data, train_label, test_size=0.25, random_state=7)

classifier = LogisticRegression()
classifier.fit(train_x, train_y)
score = classifier.score(test_x, test_y)
predict = classifier.predict(test_x)

reshape = np.array(predict).reshape(-1, 1)
vstack = np.hstack((reshape, test_y))

print()
