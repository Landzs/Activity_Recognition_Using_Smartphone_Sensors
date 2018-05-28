import os
import pandas as pd
import numpy as np
import glob

data = {'data_subjects_info': pd.read_csv('./motionsense-dataset/data_subjects_info.csv', sep = ',')}
data['A_DeviceMotion_data'] = dict()

folders = os.walk('./motionsense-dataset/A_DeviceMotion_data').next()[1]
for subfolder in folders:
    data['A_DeviceMotion_data'][subfolder] = dict()
    path = r'./motionsense-dataset/A_DeviceMotion_data/' + subfolder
    files = glob.glob(path + '/*.csv')
    for f in files:
        data['A_DeviceMotion_data'][subfolder][os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f, sep = ',')

# In DeepConvLTSM: X is nd.array with observations in rows, Y is vector of labels.
X_train = np.zeros(shape = (1,16))
Y_train = np.array('init')
X_val = np.zeros(shape = (1,16))
Y_val = np.array('init')
X_test = np.zeros(shape = (1,16))
Y_test = np.array('init')

def ConcatenatingEntries(list, X, Y):
    for subject in list:
        sens_data = np.asarray(data['A_DeviceMotion_data'][subfolder][subject])[:, 1:13]

        personal_attributes = np.asarray(data['data_subjects_info'].loc[int(subject.split('_', 1)[1]) - 1, :])[1:5]
        personal_attributes = np.array([personal_attributes, ] * sens_data.shape[0])

        X = np.append(X, np.concatenate((sens_data, personal_attributes), axis = 1), axis = 0)
        Y = np.append(Y, np.repeat(label, sens_data.shape[0]))
    return [X, Y]

for subfolder in list(data['A_DeviceMotion_data']):

    label = subfolder.split('_', 1)[0]

    train_list = list(data['A_DeviceMotion_data'][subfolder])
    val_list = [None]*5
    test_list = [None]*2
    random = sorted(np.random.choice(len(train_list), 7, replace=False), reverse=True)
    for remove in range(7):
        if remove < 5:
            val_list[remove] = train_list[random[remove]]
        else:
            test_list[remove-5] = train_list[random[remove]]
        del train_list[random[remove]]

    X_train, Y_train = ConcatenatingEntries(train_list, X_train, Y_train)
    X_val, Y_val = ConcatenatingEntries(val_list, X_val, Y_val)
    X_test, Y_test = ConcatenatingEntries(test_list, X_test, Y_test)

X_train = np.delete(X_train, 0, axis = 0)
Y_train = np.delete(Y_train, 0)
X_val = np.delete(X_val, 0, axis = 0)
Y_val = np.delete(Y_val, 0)
X_test = np.delete(X_test, 0, axis = 0)
Y_test = np.delete(Y_test, 0)

def IntegerEncode(Y):
    Y_encode = np.zeros(len(Y), dtype = int)
    for i in range(len(Y)):
        if Y[i] == 'std':
            Y_encode[i] = 0
        if Y[i] == 'dws':
            Y_encode[i] = 1
        if Y[i] == 'sit':
            Y_encode[i] = 2
        if Y[i] == 'wlk':
            Y_encode[i] = 3
        if Y[i] == 'jog':
            Y_encode[i] = 4
        if Y[i] == 'ups':
            Y_encode[i] = 5
    return Y_encode

Y_train = IntegerEncode(Y_train)
Y_val = IntegerEncode(Y_val)
Y_test = IntegerEncode(Y_test)

# Exporting
np.savetxt('Project_Data/X_train.csv', X_train)
np.savetxt('Project_Data/X_val.csv', X_val)
np.savetxt('Project_Data/X_test.csv', X_test)
np.savetxt('Project_Data/Y_train.csv', Y_train, fmt = '%10.0f')
np.savetxt('Project_Data/Y_val.csv', Y_val, fmt = '%10.0f')
np.savetxt('Project_Data/Y_test.csv', Y_test, fmt = '%10.0f')