import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn.svm import LinearSVR

def load_data_regression():
    diabetes=datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def load_data_classification():
    digits=datasets.load_digits()
    return cross_validation.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

def test_RandomForestClassifier(*data):
    X_train,X_test, Y_train,Y_test = data
    clf=ensemble.RandomForestClassifier()
    clf.fit(X_train,Y_train)
    print("Training score : %f" % (clf.score(X_train,Y_train)))
    print("Testing score : %f" % (clf.score(X_test,Y_test)))

def test_RandomForestClassifier_num(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    nums=np.arange(1,100,2)
    trains_score = []
    tests_score = []
    for i,num in enumerate(nums):
        clf=ensemble.RandomForestClassifier(n_estimators=num)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(nums, trains_score, label = 'trains score', marker='o')
    ax.plot(nums, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    fig.suptitle("RandomForestClassifier_base_classifier")
    plt.show()

def test_RandomForestClassifier_MaxDepth(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    max_depths=np.arange(1,20)
    trains_score = []
    tests_score = []
    for i,max_depth in enumerate(max_depths):
        clf=ensemble.RandomForestClassifier(max_depth=max_depth)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(max_depths, trains_score, label = 'trains score', marker='o')
    ax.plot(max_depths, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"max_depths")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    fig.suptitle("RandomForestClassifier_base_classifier")
    plt.show()

def test_RandomForestClassifier_MaxFeatures(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    MaxFeatures=np.arange(1,20)
    trains_score = []
    tests_score = []
    for i,max_feature in enumerate(MaxFeatures):
        clf=ensemble.RandomForestClassifier(max_features=max_feature)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(MaxFeatures, trains_score, label = 'trains score', marker='o')
    ax.plot(MaxFeatures, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"MaxFeatures")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    fig.suptitle("RandomForestClassifier_base_classifier")
    plt.show()


def test_RandomForestRegressor(*data):
    X_train,X_test, Y_train,Y_test = data
    clf=ensemble.RandomForestRegressor()
    clf.fit(X_train,Y_train)
    print("Training score : %f" % (clf.score(X_train,Y_train)))
    print("Testing score : %f" % (clf.score(X_test,Y_test)))

def test_RandomForestRegressor_num(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    nums=np.arange(1,100,2)
    trains_score = []
    tests_score = []
    for i,num in enumerate(nums):
        clf=ensemble.RandomForestRegressor(n_estimators=num)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(nums, trains_score, label = 'trains score', marker='o')
    ax.plot(nums, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    fig.suptitle("RandomForestRegressor_base_classifier")
    plt.show()

def test_RandomForestRegressor_MaxDepth(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    max_depths=np.arange(1,20)
    trains_score = []
    tests_score = []
    for i,max_depth in enumerate(max_depths):
        clf=ensemble.RandomForestRegressor(max_depth=max_depth)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(max_depths, trains_score, label = 'trains score', marker='o')
    ax.plot(max_depths, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"max_depths")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    fig.suptitle("RandomForestRegressor_base_classifier")
    plt.show()

def test_RandomForestRegressor_MaxFeatures(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    MaxFeatures=np.linspace(0.01,1)
    trains_score = []
    tests_score = []
    for i,max_feature in enumerate(MaxFeatures):
        clf=ensemble.RandomForestRegressor(max_features=max_feature)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(MaxFeatures, trains_score, label = 'trains score', marker='o')
    ax.plot(MaxFeatures, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"MaxFeatures")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    fig.suptitle("RandomForestRegressor_base_classifier")
    plt.show()

X_train,X_test, Y_train,Y_test = load_data_regression()
test_RandomForestRegressor(X_train,X_test, Y_train,Y_test)
test_RandomForestRegressor_MaxFeatures(X_train,X_test, Y_train,Y_test)
test_RandomForestRegressor_MaxDepth(X_train,X_test, Y_train,Y_test)
test_RandomForestRegressor_num(X_train,X_test, Y_train,Y_test)

X_train,X_test, Y_train,Y_test = load_data_classification()
test_RandomForestClassifier(X_train,X_test, Y_train,Y_test)
test_RandomForestClassifier_MaxFeatures(X_train,X_test, Y_train,Y_test)
test_RandomForestClassifier_MaxDepth(X_train,X_test, Y_train,Y_test)
test_RandomForestClassifier_num(X_train,X_test, Y_train,Y_test)

