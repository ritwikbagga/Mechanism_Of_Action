from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
import pandas as pd
from sklearn.model_selection import  train_test_split as train_test_split


def load_data():
    Train_featues = pd.read_csv('Data/train_features.csv')

    Train_targets_scored = pd.read_csv("Data/train_targets_scored.csv")
    Train_targets_nonscored =  pd.read_csv("Data/train_targets_nonscored.csv")
    Test_features = pd.read_csv("Data/test_features.csv")

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

    #categorical data
    Data = add_dummies(Data, "cp_dose")
    Data = add_dummies(Data, "cp_time")
    Data = add_dummies(Data, "cp_type")
    Data = Data.drop("sig_id" , axis= 1)
    return Data


class SeqModel(nn.Module):
    def __init__(self, num_features, num_targets):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3).
            nn.PReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(1024, num_targets)

        )

    def forward(self, x):
        self.model(x)
        return x



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    accuracy = test(model, device, train_loader)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy











def main():
    Train_featues, Train_targets_scored, Test_features = load_data()
    Train_targets_scored= Train_targets_scored.drop("sig_id" , axis= 1)   #shape (23814,207 )
    Train_featues = preprocess(Train_featues)      # shape (23814, 880)
    Test_features = preprocess(Test_features)

    train_data , valid_data, train_target, valid_target = train_test_split(Train_featues, Train_targets_scored, test_size=0.1, random_state=42)



    use_cuda= False
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 32
    device = torch.device("cuda" if use_cuda else "cpu")

    tensor_x = torch.tensor(train_data.iloc[:, :].values, device=device)
    tensor_y = torch.tensor(train_target.iloc[:, :].values, device=device)


    test_tensor_x = torch.tensor(Train_featues.iloc[:, :].values, device=device)
    test_tensor_y = torch.tensor(Test_features.iloc[:, :].values, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size)  # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x, test_tensor_y)  # create your datset
    test_loader = utils.DataLoader(test_dataset)


    model = SeqModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #add weight decay?

    train_acc_list = []
    test_acc_list = []
    epoch_list = []
    for epoch in range(NumEpochs):
        epoch_list.append(epoch)
        train_acc = train(model, device, train_loader, optimizer, epoch)
        train_acc_list.append(train_acc)
        print('\nTrain set Accuracy: {:.1f}%\n'.format(train_acc))
        test_acc = test(model, device, test_loader)
        print('\nTest set Accuracy: {:.1f}%\n'.format(test_acc))
        test_acc_list.append(test_acc)










if __name__ == '__main__':
    main()