from __future__ import print_function
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
import pandas as pd
from sklearn.model_selection import  train_test_split as train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multioutput import ClassifierChain

mdl = [

(MultiOutputClassifier(LinearDiscriminantAnalysis(solver='svd')), "MultiOutputClassifier(LinearDiscriminantAnalysis(solver='svd'))"), # 0.3xx

(MultiOutputClassifier(LinearDiscriminantAnalysis(solver='lsqr')), "MultiOutputClassifier(LinearDiscriminantAnalysis(solver='lsqr'))"), #0.3xx


#(MultiOutputClassifier(LinearDiscriminantAnalysis(solver='eigen')), "MultiOutputClassifier(LinearDiscriminantAnalysis(solver='eigen'))"), # 0.3xx
#  MultiOutputClassifier(SVC()), # 0.4005246444843297
#  MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=10, learning_rate=2)),
#  MultiOutputClassifier(AdaBoostClassifier(RandomForestClassifier())), # 0.40204335220212617
#  MultiOutputClassifier(GaussianNB()),
# ClassifierChain(GaussianNB()), #0.0009664503658704957
# #ClassifierChain(SGDClassifier(loss='perceptron')),
# ClassifierChain(KNeighborsClassifier()),
# ClassifierChain(DecisionTreeClassifier(max_depth=10)), #ACC: 0.3296976390998205,  mean(loss**2) = 0.00177
# ClassifierChain(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)))
]

















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

def get_PCA(X_train_feats, dims):

    g_pca = PCA(n_components = dims)
    g_pca=  g_pca.fit(X_train_feats)
    return g_pca





def main():

    Train_featues, Train_targets_scored, Test_features = load_data()
    Train_targets_scored = Train_targets_scored.drop("sig_id", axis=1)  # shape (23814,207 )
    Train_featues = preprocess(Train_featues)  # shape (23814, 880)
    Test_features = preprocess(Test_features)

   # train_X, test_X, train_Y, test_Y = train_test_split(Train_featues, Train_targets_scored, test_size=0.1, random_state=42)
    pca = get_PCA(Train_featues, 10)
    X_transformed = pca.transform(Train_featues)
    # print(X_transformed.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, Train_targets_scored, test_size=0.33, random_state=42)
    print("----------------------------Running Models------------")
    for m in mdl:
        print(m[1])

        print(f"For Model {m[1]}")

        m[0].fit(X_train, y_train)
        print(m[0].score(X_test, y_test))



if __name__ == '__main__':
    main()