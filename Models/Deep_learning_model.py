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

    #categorical data
    Data = add_dummies(Data, "cp_dose")
    Data = add_dummies(Data, "cp_time")
    Data = add_dummies(Data, "cp_type")
    Data = Data.drop("sig_id" , axis= 1)
    return Data


class SeqModel(nn.Module):
    def __init__(self, num_features, num_targets, total_layers=3, hidden_layer_size =1024, drop_out=0.5):
        super().__init__()

        Layers = []
        for i in range(total_layers):
            if len(Layers)==0:
                Layers.append(nn.Linear(num_features, hidden_layer_size))
                Layers.append(nn.BatchNorm1d(hidden_layer_size))
                Layers.append( nn.Dropout(drop_out))
                nn.PReLU()
            else:
                Layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                Layers.append(nn.BatchNorm1d(hidden_layer_size))
                Layers.append(nn.Dropout(drop_out))
                nn.PReLU()

        Layers.append(nn.Linear(hidden_layer_size, num_targets))
        self.model = nn.Sequential(*Layers)



    def forward(self, x):
        x= self.model(x.float())
        return x



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    Loss = test_mdl(model, device, train_loader)
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




loss_and_params = []
def tune_model(params, device, train_loader, epochs):
    model = SeqModel(879, 206,total_layers=params["total_layers"], hidden_layer_size=params["Hidden_layer_size"], drop_out=params["Dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])  # add weight decay?
    train_Loss_list = []
    epoch_list = []
    print(f"--------------current:params= {params}---------------")
    for epoch in range(epochs):

        epoch_list.append(epoch)
        train_loss = train(model, device, train_loader, optimizer, epoch)

        train_Loss_list.append(train_loss)
        if (epoch == epochs - 1):
            print('\nTrain set Loss: {}%\n'.format(train_loss))
            loss_and_params.append((train_loss, params))




def main():

    Train_featues, Train_targets_scored, Test_features = load_data()
    Train_targets_scored= Train_targets_scored.drop("sig_id" , axis= 1).values  #shape (23814,207 )
    Train_featues = preprocess(Train_featues).values    # shape (23814, 880)
    Test_features = preprocess(Test_features)

    train_x , test_x, train_y, test_y = train_test_split(Train_featues, Train_targets_scored, test_size=0.1, random_state=42)
    # print(sum(train_y)[:5])
    # breakpoint()
    use_cuda= False
    learning_rate = 1e-06
    NumEpochs = 10
    batch_size = 1024
    device = torch.device("cuda" if use_cuda else "cpu")

    tensor_x = torch.tensor(train_x.iloc[:, :].values,dtype=torch.float, device=device)
    tensor_y = torch.tensor(train_y.iloc[:, :].values,dtype=torch.float,  device=device)

    breakpoint()
    test_tensor_x = torch.tensor(test_x.iloc[:, :].values,dtype=torch.float, device=device)
    test_tensor_y = torch.tensor(test_y.iloc[:, :].values,dtype=torch.float)

    train_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size)  # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x, test_tensor_y)  # create your datset
    test_loader = utils.DataLoader(test_dataset)

    # Hyper parameter tuning
    #
    # params = {
    #
    #     "learning_rate": [10e-2, 10e-3, 10e-4, 10e-5],
    #     "Hidden_layer_size": np.linspace(16, 2048, 3, dtype=int),
    #     "Dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    #     "total_layers": [1, 2, 3, 4, 5, 6]
    # }
    # parameter_list = []
    # for lr in params["learning_rate"]:
    #     for hls in params["Hidden_layer_size"]:
    #         for Dropout in params["Dropout"]:
    #             for total_layers in params["total_layers"]:
    #                 parameter_list.append({"learning_rate": lr, "Hidden_layer_size": hls, "Dropout": Dropout, "total_layers": total_layers})
    #
    # print(len(parameter_list))
    # breakpoint()
    # for parameters in parameter_list:
    #     tune_model(parameters, device, train_loader,
    #                epochs=10)  # Training best =  (0.017691294973095257, {'lr': 1e-06, 'hidden_layer_size': 1032, 'dropout': 0.1})
    #
    # list_loss_params = sorted(loss_and_params, key=lambda x: x[0])  # sorting the list of (loss, params)
    # print(list_loss_params[:5])
    # breakpoint()
    """
    from parameter tuning we found that best parameters is lr =0.001 , hiden_layer_size= 2048 , dropout = 0.2
    'learning_rate': 0.01, 'Hidden_layer_size': 1032, 'Dropout': 0.3, 'total_layers': 1}
    """

    learning_rate= 0.01
    hidden_layer_size = 1032
    Dropout= 0.3
    total_layers = 1

    model = SeqModel(879, 206,total_layers=total_layers ,hidden_layer_size=hidden_layer_size, drop_out=Dropout ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #add weight decay?
    print(Train_featues.shape)
    print(Train_targets_scored.shape)
    print(Test_features.shape)
    breakpoint()
    model.fit(Train_featues, Train_targets_scored)


    print(sum(train_y[:5]))
    breakpoint()


    train_Loss_list = []
    test_Loss_list = []
    epoch_list = []
    for epoch in range(NumEpochs):
        epoch_list.append(epoch)
        train_loss = train(model, device, train_loader, optimizer, epoch)

        train_Loss_list.append(train_loss)
        print('\nTrain set Loss: {}%\n'.format(train_loss))

        test_loss = test_mdl(model, device, test_loader)
        print('\nTest set Loss: {}%\n'.format(test_loss))
        test_Loss_list.append(test_loss)

    #Plot train and test accuracy vs epoch
    plt.figure("Train and Test Loss vs Epoch")
    plt.plot(epoch_list, train_Loss_list, c='r', label="Train Loss")
    plt.plot(epoch_list, test_Loss_list, c='g', label="Test Loss")
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.legend(loc=0)
    plt.show()







if __name__ == '__main__':
    main()