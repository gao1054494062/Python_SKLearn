import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble
from sklearn import naive_bayes




def load_data_regression():
    diabetes=datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def load_data_classification():
    digits=datasets.load_digits()
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

def test_AdaBoostClassifier(*data):
    X_train,X_test, Y_train,Y_test = data
    clf=ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train,Y_train)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    estimates_num=len(clf.estimators_)
    X=range(1,estimates_num+1)
    ax.plot(list(X), list(clf.staged_score(X_train,Y_train)), label = 'trains score', marker='o')
    ax.plot(list(X), list(clf.staged_score(X_test,Y_test)), label = 'tests score', marker='*')
    ax.set_xlabel(r"estimator num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("AdaBoostClassifier")
    ax.legend(loc='best')
    plt.show()

def test_AdaBoostClassifier_base_classifier(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(2,1,1)  

    clf=ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train,Y_train)
    estimates_num=len(clf.estimators_)
    print("estimates_num = ", estimates_num)
    print("clf.classes = ", clf.classes_)
    print("clf.n_classes = ", clf.n_classes_)
    print("clf.estimators_weight_ = ", clf.estimator_weights_)
    print("clf.feature_importances_ = ", clf.feature_importances_)
    X=range(1,estimates_num+1)
    ax.plot(list(X), list(clf.staged_score(X_train,Y_train)), label = 'trains score', marker='o')
    ax.plot(list(X), list(clf.staged_score(X_test,Y_test)), label = 'tests score', marker='*')
    ax.set_xlabel(r"estimator num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("AdaBoostClassifier with decision")
    ax.legend(loc='lower right')

    ax=fig.add_subplot(2,1,2)
    clf=ensemble.AdaBoostClassifier(learning_rate=0.1,base_estimator=naive_bayes.GaussianNB())
    clf.fit(X_train,Y_train)
    estimates_num=len(clf.estimators_)
    X=range(1,estimates_num+1)
    ax.plot(list(X), list(clf.staged_score(X_train,Y_train)), label = 'trains score', marker='o')
    ax.plot(list(X), list(clf.staged_score(X_test,Y_test)), label = 'tests score', marker='*')
    ax.set_xlabel(r"estimator num")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("AdaBoostClassifier with naive_bayes")
    ax.legend(loc='lower right')
    plt.show()

def test_AdaBoostClassifier_learning_rate(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    learning_rates=np.linspace(0.01,1)
    trains_score = []
    tests_score = []
    for learn_rate in learning_rates:
        clf=ensemble.AdaBoostClassifier(learning_rate=learn_rate)
        clf.fit(X_train,Y_train)
        trains_score.append(clf.score(X_train,Y_train))
        tests_score.append(clf.score(X_test,Y_test))
    #huitu
    ax.plot(learning_rates, trains_score, label = 'trains score', marker='o')
    ax.plot(learning_rates, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"learning rate")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("AdaBoostClassifier with learning rate")
    ax.legend(loc='best')
    plt.show()

def test_AdaBoostClassifier_algorithm(*data):
    X_train,X_test, Y_train,Y_test = data
    fig=plt.figure()
    learning_rates=[0.01,0.1,0.5,0.9]
    algorithms=['SAMME','SAMME.R']
    trains_score = []
    tests_score = []
    for i,learn_rate in enumerate(learning_rates):
        ax=fig.add_subplot(2,2,i+1)
        for i,algorithm in enumerate(algorithms):
            clf=ensemble.AdaBoostClassifier(learning_rate=learn_rate,algorithm=algorithm)
            clf.fit(X_train,Y_train)
            estimates_num=len(clf.estimators_)
            X=range(1,estimates_num+1)
            # trains_score.append(clf.score(X_train,Y_train))
            # tests_score.append(clf.score(X_test,Y_test))
            ax.plot(list(X), list(clf.staged_score(X_train,Y_train)), label = '%s:Trains score'%algorithm, marker='o')
            ax.plot(list(X), list(clf.staged_score(X_test,Y_test)), label = '%s:Tests score'%algorithm, marker='*')
        ax.set_xlabel(r"learning rate")
        ax.set_ylabel("score")
        ax.legend(loc='best')
        ax.set_title("learning rate :%f"%learn_rate)
    fig.suptitle("AdaBoostClassifier")
    plt.show()

X_train,X_test, Y_train,Y_test = load_data_classification()
test_AdaBoostClassifier_algorithm(X_train,X_test, Y_train,Y_test)
# test_AdaBoostClassifier(X_train,X_test, Y_train,Y_test)
# test_AdaBoostClassifier_base_classifier(X_train,X_test, Y_train,Y_test)
# test_AdaBoostClassifier_learning_rate(X_train,X_test, Y_train,Y_test)