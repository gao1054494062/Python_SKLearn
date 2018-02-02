import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import neighbors
from sklearn import model_selection 

def create_regression_digits(n):
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    y[::5] += 0.5 - np.random.rand( int( n/5 ) ) 
    return model_selection.train_test_split(X, y ,test_size=0.2)

def test_KNeighborsRegressor(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = neighbors.KNeighborsRegressor()
    regr.fit(X_train,Y_train)
    print("Training score : %f" % (regr.score(X_train,Y_train)))
    print("Testing score : %f" % (regr.score(X_test,Y_test)))

def test_KNeighborsRegressor_k_w(*data):
    X_train,X_test, Y_train,Y_test = data
    Ks = np.linspace(1, Y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform','distance']

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for weight in weights:
        trains_score = []
        tests_score = []
        for k in Ks:
            regr = neighbors.KNeighborsRegressor(weights=weight, n_neighbors=k)
            regr.fit(X_train,Y_train)
            trains_score.append(regr.score(X_train,Y_train))
            tests_score.append(regr.score(X_test,Y_test))
            # print("Training score : %f" % (regr.score(X_train,Y_train)))
            # print("Testing score : %f" % (regr.score(X_test,Y_test)))
        ax.plot(Ks,trains_score,label = 'trains score: weight=%s'%weight, marker='o')
        ax.plot(Ks,tests_score,label = 'tests score: weight=%s'%weight, marker='*')
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()

def test_KNeighborsRegressor_k_p(*data):
    X_train,X_test, Y_train,Y_test = data
    Ks = np.linspace(1, Y_train.size, endpoint=False, dtype='int')
    Ps = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for p in Ps:
        trains_score = []
        tests_score = []
        for k in Ks:
            regr = neighbors.KNeighborsRegressor(p=p, n_neighbors=k)
            regr.fit(X_train,Y_train)
            trains_score.append(regr.score(X_train,Y_train))
            tests_score.append(regr.score(X_test,Y_test))
        ax.plot(Ks,trains_score,label = 'trains score: p=%d'%p, marker='o')
        ax.plot(Ks,tests_score,label = 'tests score: p=%d'%p, marker='*')
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()

X_train,X_test, Y_train,Y_test = create_regression_digits(1000)
test_KNeighborsRegressor(X_train,X_test, Y_train,Y_test)
test_KNeighborsRegressor_k_w(X_train,X_test, Y_train,Y_test)
test_KNeighborsRegressor_k_p(X_train,X_test, Y_train,Y_test)
