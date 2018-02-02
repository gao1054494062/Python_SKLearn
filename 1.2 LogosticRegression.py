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


def test_LogisticRegression(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train,Y_train)
    print("Coef : %s, intercept = %s " % (regr.coef_, regr.intercept_) )
    print("Score : %s" % (regr.score(X_test,Y_test)) )

def test_LogisticRegression_multinomial(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    regr.fit(X_train,Y_train)
    print("Coef : %s, intercept = %s" % (regr.coef_, regr.intercept_) )
    print("Score : %s" % regr.score(X_test,Y_test) )    

def test_LogisticRegression_C(*data):
    X_train,X_test, Y_train,Y_test = data
    Cs = np.logspace(-2, 4, num=100)
    score = []
    for C in Cs:
        regr = linear_model.LogisticRegression(C = C)
        regr.fit(X_train,Y_train)
        score.append(regr.score(X_test,Y_test))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,score)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LogisticRegression")
    plt.show()


X_train,X_test, Y_train,Y_test = load_iris()

# test_Linear_Regression(X_train,X_test, Y_train,Y_test)

# test_Lasso(X_train,X_test, Y_train,Y_test)
# test_Lasso_alpha(X_train,X_test, Y_train,Y_test)

# test_Ridge(X_train,X_test, Y_train,Y_test)
# test_Ridge_alpha(X_train,X_test, Y_train,Y_test)
test_LogisticRegression(X_train,X_test, Y_train,Y_test)
test_LogisticRegression_multinomial(X_train,X_test, Y_train,Y_test)
test_LogisticRegression_C(X_train,X_test, Y_train,Y_test)