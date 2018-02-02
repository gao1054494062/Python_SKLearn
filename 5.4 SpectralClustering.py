import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from sklearn import mixture

def create__datas(centers,num=100,std=0.7):
    # X1, y1=datasets.make_circles(n_samples=500, n_features=4, factor=.6,noise=.05)
    X1, y1=datasets.make_circles(n_samples=500, noise=.05)
    X2, y2 = make_blobs(n_samples=num, n_features=2, centers=centers,cluster_std=std)
    lables_true=np.concatenate((y1, y2))
    X=np.concatenate((X1, X2))
    # plt.scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()
    return X, lables_true


def plot_data(*data):
    X,lables_true=data
    lables=np.unique(lables_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbyckm'
    for i,lable in enumerate(lables):
        positon=lables_true==lable
        ax.scatter(X[positon,0], X[positon,1],label="cluster %d"%lable)
        color=colors[i%len(colors)]
    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_title("data")
    plt.show()

def test_SpectralClustering(*data):
    X,lables_true=data
    clst=cluster.SpectralClustering(n_clusters=3, gamma=100)
    pred=clst.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
    plt.show()
    print("ARI : %s" % adjusted_rand_score(lables_true,pred) )

def test_SpectralClustering_gamma(*data):
    X,lables_true=data
    gammas = (0.01,0.1,1,10,20,30,50,60,100)
    ARIs=[]
    for gamma in gammas:
        clst=cluster.SpectralClustering(gamma=gamma)
        pred=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(gammas,ARIs, marker='+')
    ax.set_xscale('log')
    ax.set_xlabel(r"$gamma$")
    ax.set_ylabel("ARI")
    
    fig.suptitle("SpectralClustering")
    plt.show()

def test_SpectralClustering_n_clusters(*data):
    X,lables_true=data
    n_clusters=range(1,10)
    ARIs=[]
    for n_cluster in n_clusters:
        clst=cluster.SpectralClustering(n_clusters=n_cluster)
        pred=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(n_clusters,ARIs, marker='+')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    
    fig.suptitle("SpectralClustering")
    plt.show()

X, lable_true = create__datas([[1.8,1.8]], 500, 0.5)
test_SpectralClustering(X, lable_true)
test_SpectralClustering_gamma(X, lable_true)
test_SpectralClustering_n_clusters(X, lable_true)