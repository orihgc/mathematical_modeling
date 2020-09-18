import os
import numpy as np

mathModel = os.path.dirname(__file__)

dict = {101: [1, 7], 102: [1, 8], 103: [1, 9], 104: [1, 10], 105: [1, 11], 106: [1, 12],
        107: [2, 7], 108: [2, 8], 109: [2, 9], 110: [2, 10], 111: [2, 11], 112: [2, 12],
        113: [3, 7], 114: [3, 8], 115: [3, 9], 116: [3, 10], 117: [3, 11], 118: [3, 12],
        119: [4, 7], 120: [4, 8], 121: [4, 9], 122: [4, 10], 123: [4, 11], 124: [4, 12],
        125: [5, 7], 126: [5, 8], 127: [5, 9], 128: [5, 10], 129: [5, 11], 130: [5, 12],
        131: [6, 7], 132: [6, 8], 133: [6, 9], 134: [6, 10], 135: [6, 11], 136: [6, 12]}

train_data = np.zeros((1, 22))

for i in range(1, 6):
    for j in range(1, 13):
        path = mathModel + "/temporary/train_data/S" + str(i) + "/S" + str(i) + "c" + str(j) + '_train_data.csv'
        csv = np.loadtxt(path, delimiter=",")
        csv_ = int(csv[0, -1])
        rowCol = dict[csv_]
        label = [2]
        for k in range(1, csv.shape[0]):
            csv_k_ = int(csv[k, -1])
            if csv_k_ in rowCol:
                label.append(1)
            else:
                label.append(0)
        hstack = np.hstack((csv, np.array(label).reshape(-1, 1)))
        train_data = np.vstack((train_data, hstack))

train_data = np.delete(train_data, 0, 0)
train_data = np.delete(train_data, -2, 1)

shape = train_data.shape

len = shape[0]
for i in range(0, len):
    if i >= len: break
    if int(train_data[i, -1]) == 2:
        train_data = np.delete(train_data, i, 0)
        i -= 1
        len -= 1

np.savetxt("train_data", train_data, delimiter=",", fmt='%f')
