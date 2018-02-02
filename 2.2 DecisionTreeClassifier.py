import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection

def create_data(n):
    np.random.seed(0)
    X = 5*np.random.rand(n,1)
    y = np.sin(X).ravel()
    noice_num = int( n / 5 )
    y[::5] += 3 * (0.5 - np.random.rand(noice_num));    
    return model_selection.train_test_split(X,y,test_size=0.25,random_state=1)

def load_iris():
    iris = datasets.load_iris();
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 0)

def test_Decision_Tree_Classifier(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = DecisionTreeClassifier(splitter = 'best')
    regr.fit(X_train,Y_train)
    print("Training score : %f" % (regr.score(X_train,Y_train)))
    print("Testing score : %f" % (regr.score(X_test,Y_test)))

def test_Decision_Tree_Classifier_Criterion(*data):
    X_train,X_test, Y_train,Y_test = data
    criterions = ['gini','entropy']
    for criterion in criterions:
        regr = DecisionTreeClassifier(criterion = criterion)
        regr.fit(X_train,Y_train)
        print("criterion %s " % criterion)
        print("Training score : %f" % (regr.score(X_train,Y_train)))
        print("Testing score : %f" % (regr.score(X_test,Y_test)))

def test_Decision_Tree_Classifier_Splitter(*data):
    X_train,X_test, Y_train,Y_test = data
    splites = ['best','random']
    for splite in splites:
        regr = DecisionTreeClassifier(splitter = splite)
        regr.fit(X_train,Y_train)
        print("splite %s " % splite)
        print("Training score : %f" % (regr.score(X_train,Y_train)))
        print("Testing score : %f" % (regr.score(X_test,Y_test)))
    
def test_Decision_Tree_Classifier_Depth(*data, maxdepth):
    X_train,X_test, Y_train,Y_test = data
    depths = np.arange(1, maxdepth)
    trains_score = []
    tests_score = []
    for depth in depths:
        regr = DecisionTreeClassifier(max_depth=depth)
        regr.fit(X_train,Y_train)
        trains_score.append(regr.score(X_train,Y_train))
        tests_score.append(regr.score(X_test,Y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,trains_score,label = 'trains score',marker='o')
    ax.plot(depths,tests_score,label = 'tests score',marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classifier")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()

from sklearn.tree import export_graphviz

X_train,X_test, Y_train,Y_test = load_iris()
test_Decision_Tree_Classifier(X_train,X_test, Y_train,Y_test)
clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)
export_graphviz(clf,"E:/Python/test/python实战/out")

test_Decision_Tree_Classifier_Criterion(X_train,X_test, Y_train,Y_test)
test_Decision_Tree_Classifier_Splitter(X_train,X_test, Y_train,Y_test)
test_Decision_Tree_Classifier_Depth(X_train,X_test, Y_train,Y_test,maxdepth=20)