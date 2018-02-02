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

def test_linearSVC(*data):
    X_train, X_test, y_train, y_test=data
    cls=svm.LinearSVC()
    cls.fit(X_train,y_train)
    print("Coefficient = %s, intercept = %s" % (cls.coef_,cls.intercept_))
    print("Score = %s" % cls.score(X_test,y_test))

def test_linearSVC_parameter(*data):
    X_train, X_test, y_train, y_test=data
    losses = ['hinge','squared_hinge']
    for loss in losses:
        cls=svm.LinearSVC(loss=loss)
        cls.fit(X_train,y_train)
        print("loss = %s" % loss)
        print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        print("Score = %s" % cls.score(X_test,y_test))
    
    penalty = ['l1','l2']
    for p in penalty:
        cls=svm.LinearSVC(penalty=p,dual=False)
        cls.fit(X_train,y_train)
        print("loss = %s" % p)
        print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        print("Score = %s" % cls.score(X_test,y_test))

    Cs = np.logspace(-2,1)
    train_score=[]
    test_score=[]
    for C in Cs:
        cls=svm.LinearSVC(C=C)
        cls.fit(X_train,y_train)
        print("loss = %s" % C)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,train_score, label='Train score',marker='+')
    ax.plot(Cs,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_title("LinearSVM")
    ax.legend(loc='best')
    plt.show()



X_train, X_test, y_train, y_test=load_data_classifier()
test_linearSVC_parameter(X_train, X_test, y_train, y_test)
# test_linearSVC_parameter(X_train, X_test, y_train, y_test)