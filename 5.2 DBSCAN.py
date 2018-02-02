import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from sklearn import mixture

def create__datas(centers,num=100,std=0.7):
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

def test_DBSCAN(*data):
    X,lables_true=data
    clst=cluster.DBSCAN(eps = 0.1)
    pred=clst.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
    plt.show()
    print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
    print("Core sample num : %s" % len(clst.core_sample_indices_))

def test_DBSCAN_epsilon(*data):
    X,lables_true=data
    epsilons = np.logspace(-1, 1.5)
    ARIs=[]
    Core_nums=[]
    for epsilon in epsilons:
        clst=cluster.DBSCAN(eps=epsilon)
        clst.fit(X)
        pred=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        Core_nums.append(len(clst.core_sample_indices_))
        # print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(epsilons,ARIs, marker='+')
    ax.set_xscale('log')
    ax.set_xlabel(r"$epsilon$")
    ax.set_ylim(0,1)
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(epsilons,Core_nums, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel(r"$epsilon$")
    ax.set_ylabel("Core_nums")
    
    fig.suptitle("DBSCAN")
    plt.show()

def test_DBSCAN_min_samples(*data):
    X,lables_true=data
    min_sampls=range(1,100)
    ARIs=[]
    Core_nums=[]
    for min_sample in min_sampls:
        clst=cluster.DBSCAN(min_samples=min_sample)
        clst.fit(X)
        pred=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        Core_nums.append(len(clst.core_sample_indices_))
        # print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(min_sampls,ARIs, marker='+')
    ax.set_xscale('log')
    ax.set_xlabel("min_samples")
    ax.set_ylim(0,1)
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(min_sampls,Core_nums, marker='o')
    ax.set_xscale('log')
    ax.set_xlabel("min_samples")
    ax.set_ylabel("Core_nums")
    
    fig.suptitle("DBSCAN")
    plt.show()

X, lable_true = create__datas([[1.8,1.8]], 100, 0.5)
# X, lable_true = create__datas([[1,1],[2,2],[1,2],[10,20]], 1000, 0.5)
# plot_data(X, lable_true)
test_DBSCAN(X, lable_true)
test_DBSCAN_epsilon(X, lable_true)
test_DBSCAN_min_samples(X, lable_true)
# test_DBSCAN_n_init(X, lable_true)