import os
import numpy as np
import pandas as pd

test_data_path = []
test_event_path = []

mathModel = os.path.dirname(__file__)
for i in range(1, 6):
    index = str(i)
    test_data_path.append(mathModel + "/data/persons/S" + index + "/S" + index + "_test_data.xlsx")
    test_event_path.append(mathModel + "/data/persons/S" + index + "/S" + index + "_test_event.xlsx")

for i in range(0, 5):
    sheetCount = 10
    if i == 1 or i == 2:
        sheetCount = 9
    for j in range(0, sheetCount):
        test_event = pd.read_excel(test_event_path[i], sheet_name=j, header=None)
        test_data = pd.read_excel(test_data_path[i], sheet_name=j, header=None)
        test = []
        for test_data_index in test_event.iloc[:, -1]:
            test.append(np.array(test_data.iloc[test_data_index - 1, :]))
        test_event_iloc_ = np.array(test_event.iloc[:, 0]).reshape(-1, 1)
        test = np.hstack((test, test_event_iloc_))
        personIndex = str(i + 1)
        charIndex = str(j + 1)
        np.savetxt("S" + personIndex + "c" + charIndex + "_test_data.csv", test,
                   delimiter=",", fmt='%f')
