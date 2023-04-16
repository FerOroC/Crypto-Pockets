from linear_fea import *
#from model import *
from visualisation import *
#from utils import *

import numpy as np
from pandas import DataFrame
import torch
import torch.nn as nn
from tensorflow.python.keras.engine.input_layer import Input
import tensorflow as tf

#---------------------------------------------------------------CONSTANTS-----------------------------------------------------------------------------------------

E = 1944*10**6 #(N/m^2)   youngs mosulus of PA12
A = 1.9635*10**-5 #0.01 #(m^2) area for a dia of bar 0.005m
I = 6.1359*10**-11 #(m^4) #second moment area for shaft

members = np.array([[1,2],
                    [2,3],
                    [3,4],
                    [4,5],
                    [5,6],
                    [6,7],
                    [7,8],
                    [8,9],
                    [9,10],
                    [10,11],
                    [11,12],
                    [12,13],
                    [13,14],
                    [14,1],   #Element N
                    [1,15],       #15_1
                    [2,15],
                    [16,15],
                    [14,15],
                    [2,16],            #16_
                    [3,16],
                    [17,16],
                    [3,17],             #17_
                    [4,17],
                    [5,17],
                    [18,17],
                    [5,18],           #18_
                    [6,18],
                    [19,18],
                    [6,19],                #19_
                    [7,19],
                    [20,19],
                    [7,20],                    #20_
                    [8,20],
                    [9,20],
                    [21,20],
                    [9,21],                         #21
                    [10,21],
                    [22,21],
                    [10,22],                              #22
                    [11,22],
                    [12,22],
                    [14,23],                        #23
                    [15,23],
                    [24,23],
                    [26,23],
                    [13,23],
                    [16,24],                        #24
                    [17,24],
                    [18,24],
                    [25,24],
                    [19,25],                #25
                    [20,25],
                    [21,25],
                    [26,25],
                    [22,26],            #26
                    [12,26],
                    [13,26]
                    ])

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

Input_Forces = []
Input_Nodes = []
Node_List = []

Axial_Forces = []
Bending_Moments = []
Nodal_Displacements = []

Euclidian_Distance = []
STD_Distance = []

Performance_Coefficient = []

LatticeGenerator(25000)

df = DataFrame({
    'Input Node Position':Input_Nodes,
    'Performance_Coefficient': Performance_Coefficient,
})

df.to_csv('Final_Dataset.csv', index=False)

Input_Nodes = np.array(df['Input Node Position'])

X_train = np.array(Input_Nodes[:19999:1])
Y_train = np.array(Performance_Coefficient[:19999:1])

print(Y_train)

X_test = np.array(Input_Nodes[19999::1])
Y_test = np.array(Performance_Coefficient[19999::1])

print(Input_Nodes[1])

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units = 52, activation = "selu", input_shape = (1,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 256, activation = "selu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 256, activation = "selu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 52, activation = "selu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 52, activation = 'selu'))

model.compile(optimizer = 'adam', loss = tf.keras.losses.MeanSquaredError(), metrics="MeanAbsoluteError")

model.summary()

model.fit(Y_train, X_train, epochs =15)

test_loss, test_accuracy = model.evaluate(Y_test,X_test)

#check if you have to send all train and test date to cuda device (probably yes)
# train_data = torch.utils.data.TensorDataset(x_train, y_train)
# train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle = True)
# val_data = torch.utils.data.TensorDataset(x_val, y_val)
# val_iter = torch.utils.data.DataLoader(val_data, batch_size)

#define loss, which should be meansquarederror, metrics mean absolute error and mape

#define optimizer and scheduler

#instantiate model

# model = NeuralNetwork()
# print("Model details:\n", model)
#
# loss_fn = nn.BCELoss()  # binary cross entropy
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# n_epochs = 50
# batch_size = 10
# save_model_path = "LatticePredictionModel.pt"

# min_loss = np.inf
# for epoch in range(n_epochs):
#     for i in range(0, len(X), batch_size):
#         Xbatch = X[i:i+batch_size]
#         y_pred = model(Xbatch)
#         ybatch = y[i:i+batch_size]
#         loss = loss_fn(y_pred, ybatch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if loss < min_loss:
#         min_loss = loss
#         torch.save(model.state_dict(), save_model_path)
#     print(f'Finished epoch {epoch}, latest loss {loss}')
#
# best_model = NeuralNetwork().to(device)
# best_model.load_state_dict(torch.load(save_model_path))
#
# best_model.load_state_dict(torch.load(save_model_path))
#
#
# l = evaluate_model(best_model, loss, test)
# MAE, RMSE, MAPE = evaluate_metric(best_model, test_iter)
# print("test loss:", l, "\nMAE:", MAE, "RMSE:", RMSE, "MAPE:", MAPE)