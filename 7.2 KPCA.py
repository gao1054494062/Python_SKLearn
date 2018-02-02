import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn import manifold

def load_data():
    iris=datasets.load_iris()
    return iris.data, iris.target
def test_KernelPCA(*data):
    X,y=data
    kernels=['rbf','poly','linear','sigmoid']
    fig=plt.figure()
    colors=((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),(0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    for i,kernel in enumerate(kernels):
        ax=fig.add_subplot(2,2,i+1)
        model=decomposition.KernelPCA(n_components=2, kernel=kernel)
        model.fit(X)
        X_r=model.transform(X)
        print("lambdas_ 1 = %s" % model.lambdas_)
        print("alphas_ 1 = %s" % model.alphas_)

        for label, color in zip(np.unique(y), colors):
            position=y==label
            ax.scatter(X_r[position,0], X_r[position,1], label="target=%d"%label,color=color)
        ax.set_xlabel("X[0]")
        ax.set_ylabel("Y[0]")
        ax.legend(loc="best")
        ax.set_title("kernel=%s"%kernel)
    plt.suptitle("KernelPCA")
    plt.show()

def test_KernelPCA_poly(*data):
    X,y=data
    fig=plt.figure()
    colors=((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),(0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    params=[ (3,1,1), (3,10,1), (3,1,10), (3,10,10), (10,1,1), (10,10,1), (10,1,10), (10,10,10) ]
    for i,(p,gamma,r) in enumerate(params):
        model=decomposition.KernelPCA(n_components=2, kernel='poly',gamma=gamma,degree=p,coef0=r)
        model.fit(X)
        X_r=model.transform(X)
        ax=fig.add_subplot(2,4,i+1)
        print("lambdas_ 1 = %s" % model.lambdas_)
        print("alphas_ 1 = %s" % model.alphas_)

        for label, color in zip(np.unique(y), colors):
            position=y==label
            ax.scatter(X_r[position,0], X_r[position,1], label="target=%d"%label,color=color)
        ax.set_xlabel("X[0]")
        ax.set_ylabel("Y[0]")
        ax.legend(loc="best")
        ax.set_title("gamma=%s,p=%s,r=%s"%(gamma,p,r))
    plt.suptitle("KernelPCA POLY")
    plt.show()

def test_KernelPCA_rbf(*data):
    X,y=data
    fig=plt.figure()
    colors=((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),(0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    gammas=[0.5,1,2,4,5,10]
    for i,gamma in enumerate(gammas):
        model=decomposition.KernelPCA(n_components=2, kernel='rbf',gamma=gamma)
        model.fit(X)
        X_r=model.transform(X)
        ax=fig.add_subplot(2,3,i+1)
        for label, color in zip(np.unique(y), colors):
            position=y==label
            ax.scatter(X_r[position,0], X_r[position,1], label="target=%d"%label,color=color)
        ax.set_xlabel("X[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("X[1]")
        ax.legend(loc="best")
        ax.set_title("gamma=%s"%(gamma))
    plt.suptitle("KernelPCA RBF")
    plt.show()

X, y=load_data()
test_KernelPCA_rbf(X, y)
test_KernelPCA_poly(X, y)
test_KernelPCA(X, y)