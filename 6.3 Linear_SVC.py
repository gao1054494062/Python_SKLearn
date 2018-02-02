import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm

def load_data_regression():
    diabetes=datasets.load_diabetes()
    return cross_validation.train_test_split(datasets.data,diabetes.target, test_size=0.2,random_state=0)

def load_data_classifier():
    iris=datasets.load_iris()
    return cross_validation.train_test_split(iris.data,iris.target, test_size=0.2,random_state=0, stratify=iris.target)

def test_linearSVR(*data):
    X_train, X_test, y_train, y_test=data
    cls=svm.LinearSVR()
    cls.fit(X_train,y_train)
    print("Coefficient = %s, intercept = %s" % (cls.coef_,cls.intercept_))
    print("Score = %s" % cls.score(X_test,y_test))

def test_linearSVR_parameter(*data):
    X_train, X_test, y_train, y_test=data
    fig=plt.figure()

    losses = ['epsilon_insensitive','squared_epsilon_insensitive']
    for loss in losses:
        cls=svm.LinearSVR(loss=loss)
        cls.fit(X_train,y_train)
        print("loss = %s" % loss)
        print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        print("Score = %s" % cls.score(X_test,y_test))
    
    train_score=[]
    test_score=[]
    epsilons = np.logspace(-2,1)
    for epsilon in epsilons:
        cls=svm.LinearSVR(epsilon=epsilon,loss='squared_epsilon_insensitive')
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("loss = %s" % epsilon)
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,2,1)
    ax.plot(epsilons,train_score, label='Train score',marker='*')
    ax.plot(epsilons,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"epsilons")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_ylim(-1,1.1)
    ax.set_title("LinearSVC")
    ax.legend(loc='best',framealpha=0.5)

    Cs = np.logspace(-2,2)
    train_score=[]
    test_score=[]
    for C in Cs:
        cls=svm.LinearSVR(epsilon=0.1, loss='squared_epsilon_insensitive', C=C)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,2,2)
    ax.plot(Cs,train_score, label='Train score',marker='*')
    ax.plot(Cs,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_ylim(-1,1.1)
    ax.set_title("LinearSVM")
    ax.legend(loc='best',framealpha=0.5)
    plt.show()



X_train, X_test, y_train, y_test=load_data_classifier()
test_linearSVR(X_train, X_test, y_train, y_test)
test_linearSVR_parameter(X_train, X_test, y_train, y_test)