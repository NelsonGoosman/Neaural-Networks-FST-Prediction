# original model, ~0.6 mse

import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import torch.optim
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import r2_score

# csv file path
path = "C:\\Users\\nelso\\OneDrive\\Documents\\FST Data Analysis (CSV).csv"

# initialize gpu for computation
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"Model Running on device: {device}")


def csv_to_dataframe(data_path, training_percent=0.8):
    # convert csv file to dataframe
    df = pd.read_csv(data_path, usecols=[i for i in range(9)])


    column_indices = [1, 2, 3, 4, 9, 10] # Airtemp, solar A, DP, WS, FruitSize

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['treatment'].replace(to_replace=['Control', 'Fogging', 'Netting', 'Fognet', 'Conventional'], value=[0, 1, 2, 3, 4], inplace=True)
    df['hours'] = df['datetime'].dt.hour / 24
    df['dayofyear'] = df['datetime'].dt.dayofyear / 365
    df.info()

    # class for scaling data
    ss_x = preprocessing.MinMaxScaler(feature_range=(-1,1))
    ss_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    # select what columns to use for x (input) and turn them into tensors
    X = df.iloc[:, column_indices]
    Y = df.iloc[:, 0]
    scale_X = ss_x.fit_transform(X.to_numpy())
    scale_Y = ss_y.fit_transform(Y.to_numpy().reshape(-1, 1))
    # randomly partition data into training and test sets (X = input, Y = output)
    train_X, test_X, train_Y, test_Y = train_test_split(scale_X, scale_Y, train_size=training_percent, random_state=43)
    # convert to 32-bit floats
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_Y = train_Y.astype('float32')
    test_Y = test_Y.astype('float32')

    # final tensors to be used for ML (to(device) = run on gpu)
    train_X = torch.from_numpy(train_X).to(device)
    test_X = torch.from_numpy(test_X).to(device)
    train_Y = torch.from_numpy(train_Y).to(device)
    test_Y = torch.from_numpy(test_Y).to(device)


    # print("Length of training data:" + str(len(train_X)))
    # print("Length of testing data:" + str(len(test_X)))
    return train_X, train_Y, test_X, test_Y, ss_y

# dataset class
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

# Network class used to define/initialize NN
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        hidden_size1 = 6
        hidden_size2 = 3

        self.hidden_layer1 = nn.Linear(input_size, hidden_size1)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        hidden1 = F.relu(self.hidden_layer1(x))
        hidden2 = F.relu(self.hidden_layer2(hidden1))
        output = self.output_layer(hidden2)

        return output

training_percent = 0.8

train_X, train_Y, test_X, test_Y, ss_y = csv_to_dataframe(path, training_percent=training_percent)

# hyperparameters
batch_size = 48
lr = .0005
loss = nn.MSELoss()
epochs = 100

num_inputs = len(train_X[0])
# initialize the dataset, then pass it to the dataloader for training
training_data = FSTDataset(train_X, train_Y)
testing_data = FSTDataset(test_X, test_Y)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# initialize the model used for testing, num inputs varies based on quantity of input data used
# but outputs will always be 1
model = Network(num_inputs, 1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0001)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=.95, weight_decay=.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
# scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=50)
best_mse = np.inf  # initialized to infinity
best_weights = None
history = []  # to keep track of loss function for graphing
training_history = []

# start timing
startTime = time.time()

# training loop
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
        outputs = outputs.view(predicted_output.size())
        train_loss = loss(predicted_output, outputs)

        # backpropagation
        train_loss.backward()

        optimizer.step()
        # scheduler.step()

    model.eval()

    with torch.no_grad():

        predicted_output = model(test_X)
        predicted_output = predicted_output.view(142)
        test_Y = test_Y.view(predicted_output.size())
        val_loss = loss(predicted_output, test_Y)
        mse = val_loss.item()
        history.append(mse)

        actual_unscaled = ss_y.inverse_transform(test_Y.cpu().numpy().reshape(-1, 1))
        predicted_output_unscaled = ss_y.inverse_transform(predicted_output.cpu().numpy().reshape(-1, 1))
        mse_unscaled = loss(torch.from_numpy(predicted_output_unscaled), torch.from_numpy(actual_unscaled))

        if mse < best_mse:
            best_mse = mse
            # best_weights = model.state_dict().copy()

        if epoch % 10 == 0:
            if epoch == 0:
                print("Initial Loss From Randomized Values " + str(round(history[epoch], 5)))
            else:
                print("Epoch " + str(epoch) + " / " + str(epochs) + " Loss: " + str(round(history[epoch], 5)))





# r2 = r2_score(test_Y_unscaled, predicted_output_unscaled)
# model.load_state_dict(best_weights)
seconds = time.time() - startTime
mins = seconds // 60
sec = int(seconds % 60)

print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

print("MSE (Unscaled): " + str(round(mse_unscaled.item(), 5)))
print("RMSE(Unscaled): " + str(np.sqrt(mse_unscaled.item())))

print('Model Execution Time: ' + str(mins) + " Minutes " + str(sec) + " seconds")

plt.plot(history)

plt.show()

