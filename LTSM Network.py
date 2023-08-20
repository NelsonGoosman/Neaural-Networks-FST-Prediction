import math

import pytorch_optimizer
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"Model Running on device: {device} using")
print()

path = "C:\\Users\\nelso\\Downloads\\FSTwFSTgt (1).csv"

#data preprocessing
def csv_dataloader(data_path):
    # accepts a path to the csv file, converts time series data to datetime type, and
    # returns input and output as numpy arrays
    df = pd.read_csv(data_path, usecols=[i for i in range(8)])


    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hours'] = df['datetime'].dt.hour / 24
    df['dayofyear'] = df['datetime'].dt.dayofyear / 365
    # df['datetime'] = df['datetime'].dt.time
    # df['treatment'].replace(to_replace=['Control', 'Fogging', 'Netting', 'Fognet', 'Conventional'],
                            # value=[0, 1, 2, 3, 4], inplace=True)
    return df


class FSTDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.output)

    def __getitem__(self, item):
        return self.input[item], self.output[item]

    @staticmethod
    def len_item(item):
        return len(item)


class FSTModel(nn.Module):
    def __init__(self, input_size, window_size, hidden_layer_size=60, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.window_size = window_size
        self.input_size = input_size

        self.layers = 2
        self.rl1_out = int(hidden_layer_size/2)
        self.rl2_out = int(self.rl1_out/2)

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=self.layers, dropout=.3)
        self.rl1 = nn.Linear(hidden_layer_size, self.rl1_out)
        self.rl2 = nn.Linear(self.rl1_out, self.rl2_out )
        self.output = nn.Linear(self.rl2_out, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.layers, self.window_size, self.hidden_layer_size).requires_grad_().to(device)

        c0 = torch.zeros(self.layers, self.window_size, self.hidden_layer_size).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        rl1 = self.rl1(out[:, -1, :])
        rl2 = self.rl2(rl1)
        predictions = self.output(rl2)

        return predictions



def create_dataset(dataset_x, dataset_y, lookback):

    X, y = [], []
    for i in range(len(dataset_x) - lookback):
        feature = dataset_x[i:i + lookback]
        target = dataset_y[i+lookback:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


best_mse = np.inf

dataframe = csv_dataloader(path)

dataframe.sort_values(by='datetime', inplace=True)
column_indices = [1, 2, 3, 4, 6, 8, 9]  # #colums to use as input parameters
X = dataframe.iloc[:, column_indices]
Y = dataframe.iloc[:, 7]
x_sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))
X = x_sc.fit_transform(X)
Y = y_sc.fit_transform(Y.to_numpy().reshape(-1, 1))
lookback= 35
X, Y = create_dataset(X, Y, lookback)

x_tr, x_te, y_tr, y_te = train_test_split(X, Y, train_size=0.8, shuffle=True)

x_train = x_tr.type(torch.float32)
x_test = x_te.type(torch.float32)
y_train = y_tr.type(torch.float32)
y_test = y_te.type(torch.float32)

# seen data
x_train = x_train.to(device)
x_test = x_test.to(device)
# unseen data
y_train = y_train.to(device)
y_test = y_test.to(device)


# num input parameters for the model
n_samples = len(x_train[0][0])
loss = nn.MSELoss()

# hyperparameters
epochs = 500
batchsize = 32
hidden_size = 20
lr = .0002
model = FSTModel(n_samples, lookback, hidden_size)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=lr, amsgrad=True, weight_decay=0.0001)
test_history = []
train_history = []

# initializing dataloaders
training_dataset = FSTDataset(x_train, y_train)
testing_dataset = FSTDataset(x_train, y_train)

training_data = DataLoader(training_dataset, batch_size=batchsize, shuffle=True)
testing_data = DataLoader(testing_dataset, batch_size=batchsize)

startTime = time.time()

for epoch in range(epochs):
    model.train()
    for batch_idx, batch in enumerate(training_data):
        # zero the optimizer gradient
        optimizer.zero_grad()

        inputs, outputs = batch
        # set input and output tensors to run on cuda (gpu)
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        # calculate predicted temperature and run on cuda
        predicted_output = model(inputs)

        predicted_output = predicted_output.to(device)
        # reshape to match predicted outputs shape
        # outputs = outputs.view(predicted_output.size())
        outputs = outputs.view(-1)
        predicted_output = predicted_output.view(outputs.shape)
        train_loss = loss(predicted_output, outputs)
        train_loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():

        actual_output = y_test
        predicted_output = model(x_test)
        val_loss = loss(predicted_output.view(-1, 1), actual_output.view(-1, 1))
        mse = val_loss.item()
        test_history.append(mse)

        # unscaled loss
        actual_unscaled = y_sc.inverse_transform(actual_output.cpu().numpy().reshape(-1, 1))
        predicted_unscaled = y_sc.inverse_transform(predicted_output.cpu().numpy().reshape(-1, 1))
        unscaled_loss = loss(torch.from_numpy(predicted_unscaled), torch.from_numpy(actual_unscaled))

        if epoch == epochs-1:
            real_vals, predicted_vals = [], []
            for item in y_sc.inverse_transform(actual_output.cpu().numpy().reshape(-1, 1)):
                real_vals.append(item)
            for item in y_sc.inverse_transform(predicted_output.cpu().numpy().reshape(-1, 1)):
                predicted_vals.append(item)

        if mse < best_mse:
            best_mse = mse
                # best_weights = model.state_dict().copy()

        if epoch == 0:
            print("Initial Loss From Randomized Values " + str(round(test_history[epoch], 5)))
        elif epoch % 10 == 0:
            print("Epoch " + str(epoch) + " / " + str(epochs) + " Loss: " + str(round(test_history[epoch], 5)), end=" ")
            print(f"(Unscaled: {unscaled_loss.item()})")



seconds = time.time() - startTime
mins = seconds // 60
sec = int(seconds % 60)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
print(f"MSE Unscaled: {unscaled_loss.item()}")
print(f"RMSE Unscaled: {math.sqrt(unscaled_loss.item())}")
print('Model Execution Time: ' + str(mins) + " Minutes " + str(sec) + " seconds")

plt.subplot(1, 2, 1)
plt.plot(real_vals, color='red', label='actual')
plt.plot(predicted_vals, color='blue', label='predicted')
plt.title("Real (red) vs predicted (Blue) values")
plt.subplot(1, 2, 2)
plt.plot(test_history)
plt.title("Testing loss")
plt.show()
print("Predicted")
