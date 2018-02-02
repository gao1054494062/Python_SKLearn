import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from sklearn import mixture

def create__datas(centers,num=100,std=0.7):
    X1, y1=datasets.make_circles(n_samples=2000, noise=.05)
    X2, y2 = make_blobs(n_samples=num, n_features=2, centers=centers,cluster_std=std)
    lables_true=np.concatenate((y1, y2))
    X=np.concatenate((X1, X2))
    # X, lables_true = make_blobs(n_samples=num, centers=centers,cluster_std=std)
    return X, lables_true

def test_GaussianMixture(*data):
    X,lables_true=data
    # clst=mixture.GaussianMixture()
    clst=mixture.GaussianMixture(n_components=3, covariance_type="spherical")
    clst.fit(X)
    pred=clst.predict(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
    plt.show()
    print("ARI : %s" % adjusted_rand_score(lables_true,pred) )

def test_GaussianMixture_n_componets(*data):
    X,lables_true=data
    n_componets=range(1,50)
    ARIs=[]
    for n_componet in n_componets:
        clst=mixture.GaussianMixture(n_components=n_componet)
        clst.fit(X)
        pred=clst.predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        # plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
        # plt.show()
        # print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(n_componets,ARIs, marker='+')
    ax.set_xlabel("n_componets")
    ax.set_ylabel("ARI")
    
    fig.suptitle("GaussianMixture")
    plt.show()

def test_GaussianMixture_cov_type(*data):
    X,lables_true=data
    markers="*+_."
    cov_types=['spherical','tied','diag','full']
    nums=range(1,50)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for i, cov_type in enumerate(cov_types):
        ARIs=[]
        for num in nums:
            clst=mixture.GaussianMixture(n_components=num, covariance_type=cov_type)
            clst.fit(X)
            pred=clst.predict(X)
            ARIs.append(adjusted_rand_score(lables_true,pred))
            # plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
            # plt.show()
        
            # print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
            # print("Sum center distance : %s" % clst.inertia_)
        ax.plot(nums,ARIs,marker=markers[i],label="covariance_type:%s " % cov_type)
    
    ax.set_xlabel("covariance_type")
    ax.set_ylabel("ARI")
    fig.suptitle("GaussianMixture")
    plt.show()

X, lable_true = create__datas([[1,1],[2,2],[1,2],[3,4]], 1000, 0.5)
print("X = ", X.shape)
print("lable_true = ", lable_true.shape)
test_GaussianMixture(X, lable_true)
test_GaussianMixture_cov_type(X, lable_true)
test_GaussianMixture_n_componets(X, lable_true)