import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn import manifold

def load_data():
    iris=datasets.load_iris()
    return iris.data, iris.target
def test_PCA(*data):
    X,y=data
    model=decomposition.PCA(n_components=3)
    model.fit(X)
    print("n_components_ 1 = %s" % model.n_components_)
    print("explained_variance_ 1 = %s" % model.explained_variance_)
    print("explained variance ratio 1 = %s" % str(model.explained_variance_ratio_))

    X_r=model.transform(X)
    model_r=decomposition.PCA(n_components=2)
    model_r.fit(X)
    print("n_components_ 2 = %s" % model.n_components_)
    print("explained_variance_ = %s" % model_r.explained_variance_)
    print("explained variance ratio 2 = %s" % str(model_r.explained_variance_ratio_))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),(0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    for label, color in zip(np.unique(y), colors):
        position=y==label
        ax.scatter(X_r[position,0], X_r[position,1], label="target=%d"%label,color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()

def test_PCA_mle(*data):
    X,y=data
    model=decomposition.PCA(n_components='mle')
    model.fit(X)
    print("n_components_ 3 = %s" % model.n_components_)
    print("explained_variance_ 3 = %s" % model.explained_variance_)
    print("explained variance ratio 3 = %s" % str(model.explained_variance_ratio_))

    model=decomposition.PCA(n_components=0.9)
    model.fit(X)
    print("n_components_ 4 = %s" % model.n_components_)
    print("explained_variance_ 4 = %s" % model.explained_variance_)
    print("explained variance ratio 4 = %s" % str(model.explained_variance_ratio_))

X, y=load_data()
test_PCA_mle(X, y)
test_PCA(X, y)