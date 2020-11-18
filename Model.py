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


def load_data():
    Train_featues = pd.read_csv("../Data/train_features.csv")
    Train_targets_scored = pd.read_csv("../Data/train_features.csv")


def preprocess:
    def add_dummies(df, col):
