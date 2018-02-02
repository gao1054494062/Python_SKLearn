import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation

def create_data(n):
    np.random.seed(0)
    X = 5*np.random.rand(n,1)
    y = np.sin(X).ravel()
    noice_num = int( n / 5 )
    y[::5] += 3 * (0.5 - np.random.rand(noice_num));    
    return cross_validation.train_test_split(X,y,test_size=0.25,random_state=1)

def test_Decision_Tree_Regression(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = DecisionTreeRegressor(splitter = 'best')
    regr.fit(X_train,Y_train)
    print("Training score : %f" % (regr.score(X_train,Y_train)))
    print("Testing score : %f" % (regr.score(X_test,Y_test)))
    ##绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    X = np.arange(0,5.0,0.01)[:,np.newaxis]
    Y = regr.predict(X)
    ax.scatter(X_train,Y_train, label="train sample", c='g')
    ax.scatter(X_test,Y_test, label="test sample", c='r')
    ax.plot(X,Y,label = 'predict', linewidth=2,alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regressor")
    ax.legend(framealpha=0.5)
    plt.show()

def test_Decision_Tree_Regression_Split(*data):
    X_train,X_test, Y_train,Y_test = data
    splites = ['best','random']
    for splite in splites:
        regr = DecisionTreeRegressor(splitter = splite)
        regr.fit(X_train,Y_train)
        print("Splitter %s " % splite)
        print("Training score : %f" % (regr.score(X_train,Y_train)))
        print("Testing score : %f" % (regr.score(X_test,Y_test)))
    
def test_Decision_Tree_Regression_Depth(*data, maxdepth):
    X_train,X_test, Y_train,Y_test = data
    depths = np.arange(1, maxdepth)
    trains_score = []
    tests_score = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train,Y_train)
        trains_score.append(regr.score(X_train,Y_train))
        tests_score.append(regr.score(X_test,Y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,trains_score,label = 'trains score')
    ax.plot(depths,tests_score,label = 'tests score')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regressor")
    ax.legend(framealpha=0.5)
    plt.show()

X_train,X_test, Y_train,Y_test = create_data(100)

# test_Decision_Tree_Regression(X_train,X_test, Y_train,Y_test)
# test_Decision_Tree_Regression_Split(X_train,X_test, Y_train,Y_test)

test_Decision_Tree_Regression_Depth(X_train,X_test, Y_train,Y_test,maxdepth=20)