from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
import pandas as pd
from sklearn.model_selection import  train_test_split as train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

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




def main():



if __name__ == '__main__':
    main()