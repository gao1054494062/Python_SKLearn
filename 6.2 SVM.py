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

def test_SVC_kernel(*data):
    X_train, X_test, y_train, y_test=data
    kernels = ['linear','poly','rbf','sigmoid']
    for kernel in kernels:
        cls=svm.SVC(kernel=kernel)
        cls.fit(X_train,y_train)
        print("cls.support_ = %s, cls.support_vectors_ = %s" % (cls.support_,cls.support_vectors_))
        # print("Coefficient = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        print("Score = %s" % cls.score(X_test,y_test))

def test_SVC_rbf(*data):
    X_train, X_test, y_train, y_test=data
    fig=plt.figure()
    
    gammas = range(1,20)
    train_score=[]
    test_score=[]
    for gamma in gammas:
        cls=svm.SVC(kernel='rbf', degree=gamma)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,1,1)
    ax.plot(degrees,train_score, label='Train score',marker='+')
    ax.plot(degrees,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"degree")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_yslim(0,1)
    ax.set_title("rbf SVM")
    ax.legend(loc='best')
    plt.show()

def test_SVC_sigmoid(*data):
    X_train, X_test, y_train, y_test=data
    fig=plt.figure()
    
    gammas = np.logspace(-2,1)
    train_score=[]
    test_score=[]
    for gamma in gammas:
        cls=svm.SVC(kernel='sigmoid', degree=gamma,coef0=0)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,1,1)
    ax.plot(degrees,train_score, label='Train score',marker='+')
    ax.plot(degrees,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"degree")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_yslim(0,1)
    ax.set_title("sigmoid SVM")
    ax.legend(loc='best')

    rs = np.linspace(0,5)
    train_score=[]
    test_score=[]
    for r in rs:
        cls=svm.SVC(kernel='poly', gamma=10, degree=3,coef0=r)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,3,3)
    ax.plot(rs,train_score, label='Train score',marker='+')
    ax.plot(rs,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"r")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_yslim(0,1)
    ax.set_title("poly SVM")
    ax.legend(loc='best')
    plt.show()
    plt.show()

def test_SVC_poly(*data):
    X_train, X_test, y_train, y_test=data
    fig=plt.figure()
    
    degrees = range(1,20)
    train_score=[]
    test_score=[]
    for degree in degrees:
        cls=svm.SVC(kernel='poly', degree=degree)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,3,1)
    ax.plot(degrees,train_score, label='Train score',marker='+')
    ax.plot(degrees,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"degree")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_yslim(0,1)
    ax.set_title("poly SVM")
    ax.legend(loc='best')

    gammas = range(1,100)
    train_score=[]
    test_score=[]
    for gamma in gammas:
        cls=svm.SVC(kernel='poly', gamma=gamma, degree=3)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,3,2)
    ax.plot(gammas,train_score, label='Train score',marker='+')
    ax.plot(gammas,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"gamma")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_yslim(0,1)
    ax.set_title("poly SVM")
    ax.legend(loc='best')

    rs = range(0,100)
    train_score=[]
    test_score=[]
    for r in rs:
        cls=svm.SVC(kernel='poly', gamma=10, degree=3,coef0=r)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))
        # print("Coef = %s, intercept = %s" % (cls.coef_,cls.intercept_))
        # print("Score = %s" % cls.score(X_test,y_test))
    ax=fig.add_subplot(1,3,3)
    ax.plot(rs,train_score, label='Train score',marker='+')
    ax.plot(rs,test_score, label='Test score',marker='+')
    ax.set_xlabel(r"r")
    ax.set_ylabel(r"Score")    
    ax.set_xscale('log')
    ax.set_yslim(0,1)
    ax.set_title("poly SVM")
    ax.legend(loc='best')
    plt.show()



X_train, X_test, y_train, y_test=load_data_classifier()
test_SVC_kernel(X_train, X_test, y_train, y_test)
test_SVC_poly(X_train, X_test, y_train, y_test)
test_SVC_rbf(X_train, X_test, y_train, y_test)
test_SVC_sigmoid(X_train, X_test, y_train, y_test)