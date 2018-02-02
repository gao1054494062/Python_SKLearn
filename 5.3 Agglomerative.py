import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from sklearn import mixture

def create__datas(centers,num=1000,std=0.7):
    X1, y1=datasets.make_circles(n_samples=500, factor=.6,noise=.05)
    X2, y2 = make_blobs(n_samples=num, n_features=2, centers=centers,cluster_std=std)
    lables_true=np.concatenate((y1, y2))
    X=np.concatenate((X1, X2))
    plt.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()
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

def test_AgglomerativeClustering(*data):
    X,lables_true=data
    clst=cluster.AgglomerativeClustering(n_clusters=5,linkage='average')
    pred=clst.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
    plt.show()
    print("ARI : %s" % adjusted_rand_score(lables_true,pred) )

def test_AgglomerativeClustering_nclusters(*data):
    X,lables_true=data
    nclusters = range(1,50)
    ARIs=[]
    for ncluster in nclusters:
        clst=cluster.AgglomerativeClustering(n_clusters=ncluster)
        clst.fit(X)
        pred=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        # print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(nclusters,ARIs, marker='+')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

def test_AgglomerativeClustering_linkage(*data):
    X,lables_true=data
    markers="*+."
    nums=range(1,50)
    linkages=['ward','complete','average']
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for i, linkage in enumerate(linkages):
        ARIs=[]
        for num in nums:
            clst=cluster.AgglomerativeClustering(n_clusters=num, linkage=linkage)
            pred=clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(lables_true,pred))
        ax.plot(nums, ARIs, marker=markers[i], label="linkage : %s"%linkage)
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc='best')
    fig.suptitle("AgglomerativeClustering")
    plt.show()


X, lable_true = create__datas([[1.8,1.8]], 500, 0.5)
# X, lable_true = create__datas([[1,1],[2,2],[1,2],[10,20]], 1000, 0.5)
# plot_data(X, lable_true)
test_AgglomerativeClustering(X, lable_true)
test_AgglomerativeClustering_linkage(X, lable_true)
test_AgglomerativeClustering_nclusters(X, lable_true)
