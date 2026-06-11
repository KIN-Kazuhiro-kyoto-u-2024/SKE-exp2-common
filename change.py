import numpy as np

data = np.load("data.npy",allow_pickle=True)

np.savetxt(data.csv ,data ,delimiter=",")




print(data)