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

def test_GradientBoostingClassifier(*data):
    X_train,X_test, Y_train,Y_test = data
    clf=ensemble.GradientBoostingClassifier()
    clf.fit(X_train,Y_train)
    print("Training score : %f" % (clf.score(X_train,Y_train)))
    print("Testing score : %f" % (clf.score(X_test,Y_test)))

def test_GradientBoostingClassifier_num(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    labels=['Decision Tree Regressor','linear SVR Regressor']
    regrs=[ensemble.GradientBoostingClassifier(),ensemble.GradientBoostingClassifier(base_estimator=LinearSVR(epsilon=0.01,C=100))]
    trains_score = []
    tests_score = []
    for i,regr in enumerate(regrs):
        ax=fig.add_subplot(2,1,i+1)
        regr.fit(X_train,Y_train)
        estimates_num=len(regr.estimators_)
        X=range(1,estimates_num+1)
        ax.plot(list(X), list(regr.staged_score(X_train,Y_train)), label = 'Trains score', marker='o')
        ax.plot(list(X), list(regr.staged_score(X_test,Y_test)), label = 'Tests score', marker='*')
        ax.set_xlabel(r"learning rate")
        ax.set_ylabel("score")
        ax.set_ylim(-1, 1.0)
        ax.legend(loc='best')
    fig.suptitle("GradientBoostingClassifier_base_classifier")
    plt.show()

def test_GradientBoostingClassifier_loss(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    losses=['linear','square','exponential']
    trains_score = []
    tests_score = []
    for i,loss in enumerate(losses):
        ax=fig.add_subplot(2,1,i+1)
        clf=ensemble.GradientBoostingClassifier(loss=loss,n_estimators=30)
        clf.fit(X_train,Y_train)
        estimates_num=len(clf.estimators_)
        X=range(1,estimates_num+1)
        ax.plot(list(X), list(clf.staged_score(X_train,Y_train)), label = '%s:Trains score'%loss, marker='o')
        ax.plot(list(X), list(clf.staged_score(X_test,Y_test)), label = '%s:Tests score'%loss, marker='*')
        ax.set_xlabel(r"estimator num")
        ax.set_ylabel("score")
        ax.set_ylim(0,1)
        ax.set_title("GradientBoostingClassifier with loss")
    ax.legend(loc='best')
    plt.show()

def test_GradientBoostingClassifier_learning_rate(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    learning_rates=np.linspace(0.01,1)
    algorithms=['SAMME','SAMME.R']
    trains_score = []
    tests_score = []
    ax=fig.add_subplot(1,1,1)
    print("learning_rates = ",enumerate(learning_rates))
    for i,learn_rate in enumerate(learning_rates):
        clf=ensemble.GradientBoostingClassifier(learning_rate=learn_rate,n_estimators=500)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    ax.plot(learning_rates, trains_score, label = 'trains score', marker='o')
    ax.plot(learning_rates, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"learning rate")
    ax.set_ylabel("score")
    ax.legend(loc='best')
    ax.set_title("GradientBoostingClassifier with learning rate")
    plt.show()

X_train,X_test, Y_train,Y_test = load_data_classification()
test_GradientBoostingClassifier_learning_rate(X_train,X_test, Y_train,Y_test)
test_GradientBoostingClassifier_base_classifier(X_train,X_test, Y_train,Y_test)
test_GradientBoostingClassifier_loss(X_train,X_test, Y_train,Y_test)
test_GradientBoostingClassifier(X_train,X_test, Y_train,Y_test)