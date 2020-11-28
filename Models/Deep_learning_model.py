from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split


def load_data():
    Train_featues = pd.read_csv('../Data/train_features.csv')

    Train_targets_scored = pd.read_csv("../Data/train_targets_scored.csv")
    Train_targets_nonscored =  pd.read_csv("../Data/train_targets_nonscored.csv")
    Test_features = pd.read_csv("../Data/test_features.csv")

    #return other files

    return Train_featues, Train_targets_scored , Test_features


def preprocess(Data):
    def add_dummies(df, col):
        hot_vector = pd.get_dummies(df[col])
        hot_vector_col = [f"{col}_{c}" for c in hot_vector.columns]
        hot_vector.columns = hot_vector_col
        df = df.drop(col, axis=1)
        df = df.join(hot_vector)
        return df

    # categorical data
    Data = add_dummies(Data, "cp_dose")
    Data = add_dummies(Data, "cp_time")
    Data = add_dummies(Data, "cp_type")
    Data = Data.drop("sig_id", axis=1)
    return Data


class SeqModel(nn.Module):

    def __init__(self, num_features, num_targets, batch_size):
        super().__init__()
        self.layer1_size = 256
        self.layer2_size = 256

        self.num_features = num_features
        self.num_targets = num_targets
        self.batch_size = batch_size
        self.layer1 = nn.Linear(num_features, self.layer1_size)
        self.norm = nn.BatchNorm1d(self.layer1_size)
        self.drop1 = nn.Dropout(0.5)
        self.prelu = nn.ReLU()
        self.hiddenlinear = nn.Linear(self.layer1_size, self.layer2_size)
        self.norm2 = nn.BatchNorm1d(self.layer2_size)
        self.dropout2 = nn.Dropout(0.5)
        self.final = nn.Linear(self.layer2_size, num_targets)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.drop1(x)
        x = self.prelu(x)
        x = self.hiddenlinear(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.prelu(x)
        x = self.final(x)

        return x


def train_mdl(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if batch_idx % 10 == 0:
            print(output)
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:  # Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    Loss = test_mdl(model, device, train_loader)
    # print(f"train loss = {Loss}")
    return Loss


def test_mdl(model, device, test_loader):
    model.eval()
    loss_arr = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.BCEWithLogitsLoss()(output, target)
            loss_arr.append(loss.item())

    return np.mean(loss_arr)


def main():
    Train_featues, Train_targets_scored, Test_features = load_data()
    print(Train_featues.shape)
    Train_targets_scored = Train_targets_scored.drop("sig_id", axis=1)  # shape (23814,207 )
    Train_featues = preprocess(Train_featues)  # shape (23814, 880)
    print(Train_featues.shape)

    print(Test_features.shape)
    Test_features = preprocess(Test_features)
    print(Test_features.shape)

    train_data, valid_data, train_target, valid_target = train_test_split(Train_featues, Train_targets_scored,
                                                                          test_size=0.1, random_state=42)

    use_cuda = True
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 1024
    device = torch.device("cuda" if use_cuda else "cpu")

    tensor_x = torch.tensor(train_data.iloc[:, :].values, dtype=torch.float, device=device)
    tensor_y = torch.tensor(train_target.iloc[:, :].values, dtype=torch.float, device=device)
    tensor_submit = torch.tensor(Test_features.iloc[:, :].values, dtype=torch.float, device=device)

    test_tensor_x = torch.tensor(valid_data.iloc[:, :].values, dtype=torch.float, device=device)
    test_tensor_y = torch.tensor(valid_target.iloc[:, :].values, dtype=torch.float)

    train_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, drop_last=True)  # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x, test_tensor_y)  # create your datset
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    # print(train_data.shape)
    # print(train_target.shape)
    # breakpoint()
    model = SeqModel(879, 206, batch_size)  # 879 -> 876
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # add weight decay?
    #     optimizer = torch.optim.Adadelta(model.parameters(), lr=10, rho=0.9, eps=1e-06, weight_decay=0.01)
    train_Loss_list = []
    test_Loss_list = []
    epoch_list = []
    for epoch in range(NumEpochs):
        epoch_list.append(epoch)
        train_loss = train_mdl(model, device, train_loader, optimizer, epoch)
        train_Loss_list.append(train_loss)
        print(f'\nTrain set Loss: {train_loss}')
        test_loss = test_mdl(model, device, test_loader)
        print(f'Test set Loss: {test_loss}\n')
        test_Loss_list.append(test_loss)

    # Plot train and test accuracy vs epoch
    plt.figure("Train and Test Loss vs Epoch")
    plt.plot(epoch_list, train_Loss_list, c='r', label="Train Loss")
    plt.plot(epoch_list, test_Loss_list, c='g', label="Test Loss")
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.legend(loc=0)
    plt.show()

    return model, tensor_submit, Test_features


if __name__ == '__main__':
    main()