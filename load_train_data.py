import os
import numpy as np
import pandas as pd


train_data_path = []
train_event_path = []

mathModel = os.path.dirname(__file__)
for i in range(1, 6):
    index = str(i)
    train_data_path.append(mathModel + "/data/persons/S" + index + "/S" + index + "_train_data.xlsx")
    train_event_path.append(mathModel + "/data/persons/S" + index + "/S" + index + "_train_event.xlsx")

for i in range(0, 5):
    for j in range(0, 12):
        train_event = pd.read_excel(train_event_path[i], sheet_name=j, header=None)
        train_data = pd.read_excel(train_data_path[i], sheet_name=j, header=None)
        train = []
        for train_data_index in train_event.iloc[:, -1]:
            train.append(np.array(train_data.iloc[train_data_index - 1, :]))
        train_event_iloc_ = np.array(train_event.iloc[:, 0]).reshape(-1, 1)
        train = np.hstack((train, train_event_iloc_))
        personIndex = str(i + 1)
        charIndex = str(j + 1)
        np.savetxt("S" + personIndex + "c" + charIndex + "_train_data.csv", train,
                   delimiter=",", fmt='%f')
