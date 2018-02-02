import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import model_selection

def load_sets():
    diabetes = datasets.load_diabetes();
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size = 0.25, random_state = 0)

def test_Linear_Regression(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train,Y_train)
    pred = regr.predict(X_test)
    print("Coef : %s, intercept = %.2f " % (regr.coef_, regr.intercept_) )
    print("Residual sum of squares: %.2f " % (np.mean(pred - Y_test)**2) )
    print("Score : %s" % (regr.score(X_test,Y_test)) )


def test_Ridge(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = linear_model.Ridge(alpha = 1)
    regr.fit(X_train,Y_train)
    pred = regr.predict(X_test)
    print("Coef : %s, intercept = %.2f " % (regr.coef_, regr.intercept_) )
    print("Residual sum of squares: %.2f " % (np.mean(pred - Y_test)**2) )
    print("Score : %s" % (regr.score(X_test,Y_test)) )

def test_Ridge_alpha(*data):
    X_train,X_test, Y_train,Y_test = data
    alphas = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    score = []
    for i,alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha = alpha)
        regr.fit(X_train,Y_train)
        score.append(regr.score(X_test,Y_test))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,score)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    #plt.show()

def test_Lasso(*data):
    X_train,X_test, Y_train,Y_test = data
    regr = linear_model.Lasso(alpha = 0.1)
    regr.fit(X_train,Y_train)
    pred = regr.predict(X_test)
    print("Coef : %s, intercept = %.2f " % (regr.coef_, regr.intercept_) )
    print("Residual sum of squares: %.2f " % (np.mean(pred - Y_test)**2) )
    print("Score : %s" % (regr.score(X_test,Y_test)) )

def test_Lasso_alpha(*data):
    X_train,X_test, Y_train,Y_test = data
    alphas = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores = []
    for i,alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha = alpha)
        regr.fit(X_train,Y_train)
        scores.append(regr.score(X_test,Y_test))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_xscale('log')
    ax.set_title("Ridge")
    #plt.show()



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def test_ElasticNet_alpha_rho(*data):
    X_train,X_test, Y_train,Y_test = data
    alphas = np.logspace(-2,2)
    rhos = np.linspace(0.01,1)
    scores = []
    for alpha in alphas:
    	for rho in rhos:
            regr = linear_model.ElasticNet(alpha = alpha, l1_ratio = rho)
            regr.fit(X_train,Y_train)
            scores.append(regr.score(X_test,Y_test))

    alphas,rhos = np.meshgrid(alphas,rhos)
    scores = np.array(scores).reshape(alphas.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores,rstride=1, cstride=1, cmap=cm.jet,linewidth = 0, antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()


X_train,X_test, Y_train,Y_test = load_sets()

# test_Linear_Regression(X_train,X_test, Y_train,Y_test)

# test_Lasso(X_train,X_test, Y_train,Y_test)
# test_Lasso_alpha(X_train,X_test, Y_train,Y_test)

# test_Ridge(X_train,X_test, Y_train,Y_test)
# test_Ridge_alpha(X_train,X_test, Y_train,Y_test)
test_ElasticNet_alpha_rho(X_train,X_test, Y_train,Y_test)