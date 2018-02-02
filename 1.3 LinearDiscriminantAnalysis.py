import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection

def load_iris():
    iris = datasets.load_iris();
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 0)


def test_Linear_Discrininant(*data):
    X_train,X_test, Y_train,Y_test = data
    linearDA = discriminant_analysis.LinearDiscriminantAnalysis()
    linearDA.fit(X_train,Y_train)
    print("Coef : %s, intercept = %s " % (linearDA.coef_, linearDA.intercept_) )
    print("Score : %s" % (linearDA.score(X_test,Y_test)) )

def plot_LDA():
	

X_train,X_test, Y_train,Y_test = load_iris()

test_Linear_Discrininant(X_train,X_test, Y_train,Y_test)