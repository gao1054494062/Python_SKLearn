import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import naive_bayes
from sklearn import model_selection

def load_digits():
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

def show_digits():
    digits = datasets.load_digits()
    fit = plt.figure()
    # print("vector from images 0 : ",digits[0])
    for i in range(25):
        ax = fit.add_subplot(5,5,i+1)
        ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

def test_GaussianNB(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = naive_bayes.GaussianNB()
    regr.fit(X_train,Y_train)
    print("Training score : %f" % (regr.score(X_train,Y_train)))
    print("Testing score : %f" % (regr.score(X_test,Y_test)))
    print("regr.class_prior_ : %s" % regr.class_prior_)
    print("regr.class_count_ : %s" % regr.class_count_)


def test_MultinomialNB(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = naive_bayes.MultinomialNB()
    regr.fit(X_train,Y_train)
    print("Training score : %f" % (regr.score(X_train,Y_train)))
    print("Testing score : %f" % (regr.score(X_test,Y_test)))

def test_MultinomialNB_alpha(*data):
    X_train,X_test, Y_train,Y_test = data
    alphas = np.logspace(-2,5,num=200)
    trains_score = []
    tests_score = []
    for alpha in alphas:
        regr = naive_bayes.MultinomialNB()
        regr.fit(X_train,Y_train)
        trains_score.append(regr.score(X_train,Y_train))
        tests_score.append(regr.score(X_test,Y_test))
    #huitu
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, trains_score, label = 'trains score', marker='o')
    ax.plot(alphas, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"$alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("MultinomialNB")
    ax.legend(loc='best')
    ax.set_xscale("log")
    plt.show()

def test_BernoulliNB(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = naive_bayes.BernoulliNB()
    regr.fit(X_train,Y_train)
    print("Training score : %f" % (regr.score(X_train,Y_train)))
    print("Testing score : %f" % (regr.score(X_test,Y_test)))

def test_BernoulliNB_alpha(*data):
    X_train,X_test, Y_train,Y_test = data
    alphas = np.logspace(-2,5,num=200)
    trains_score = []
    tests_score = []
    for alpha in alphas:
        regr = naive_bayes.BernoulliNB()
        regr.fit(X_train,Y_train)
        trains_score.append(regr.score(X_train,Y_train))
        tests_score.append(regr.score(X_test,Y_test))
    #huitu
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, trains_score, label = 'trains score', marker='o')
    ax.plot(alphas, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel(r"$alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("BernoulliNB")
    ax.legend(loc='best')
    ax.set_xscale("log")
    plt.show()

def test_BernoulliNB_binarize(*data):
    X_train,X_test, Y_train,Y_test = data
    min_x = min(np.min(X_train.ravel()),np.min(X_test.ravel())) - 0.1
    max_x = max(np.max(X_train.ravel()),np.max(X_test.ravel())) + 0.1
    binarizes = np.linspace(min_x, max_x, endpoint=True, num=100)
    trains_score = []
    tests_score = []
    for binarize in binarizes:
        regr = naive_bayes.BernoulliNB(binarize=binarize)
        regr.fit(X_train,Y_train)
        trains_score.append(regr.score(X_train,Y_train))
        tests_score.append(regr.score(X_test,Y_test))
    #huitu
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(binarizes, trains_score, label = 'trains score', marker='o')
    ax.plot(binarizes, tests_score, label = 'tests score', marker='*')
    ax.set_xlabel("binarize")
    ax.set_ylabel("score")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("BernoulliNB")
    ax.legend(loc='best')
    ax.set_xscale("log")
    plt.show()

# show_digits()
X_train,X_test, Y_train,Y_test = load_digits()
test_GaussianNB(X_train,X_test, Y_train,Y_test)
test_MultinomialNB(X_train,X_test, Y_train,Y_test)
test_MultinomialNB_alpha(X_train,X_test, Y_train,Y_test)
test_BernoulliNB(X_train,X_test, Y_train,Y_test)
test_BernoulliNB_alpha(X_train,X_test, Y_train,Y_test)
test_BernoulliNB_binarize(X_train,X_test, Y_train,Y_test)

