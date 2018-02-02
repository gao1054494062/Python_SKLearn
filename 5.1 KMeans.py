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

def test_KMeans(*data):
    X,lables_true=data
    clst=cluster.KMeans(init='k-means++')
    pred=clst.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=pred)
    plt.show()
    print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
    print("Sum center distance : %s" % clst.inertia_)

def test_KMeans_nclusters(*data):
    X,lables_true=data
    nums=range(1,50)
    ARIs=[]
    Distances=[]
    for num in nums:
        clst=cluster.KMeans()
        clst.fit(X)
        pred=clst.predict(X)
        ARIs.append(adjusted_rand_score(lables_true,pred))
        Distances.append(clst.inertia_)
        # print("ARI : %s" % adjusted_rand_score(lables_true,pred) )
        # print("Sum center distance : %s" % clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs, marker='+')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances, marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("KMeans")
    plt.show()

def test_KMeans_n_init(*data):
    X,lables_true=data
    nums=range(1,50)
    ARIs_k=[]
    Distances_k=[]
    ARIs_r=[]
    Distances_r=[]
    for num in nums:
        clst=cluster.KMeans(n_init=num,init='k-means++')
        clst.fit(X)
        pred=clst.predict(X)
        ARIs_k.append(adjusted_rand_score(lables_true,pred))
        Distances_k.append(clst.inertia_)

        clst=cluster.KMeans(n_init=num,init='random')
        clst.fit(X)
        pred=clst.predict(X)
        ARIs_r.append(adjusted_rand_score(lables_true,pred))
        Distances_r.append(clst.inertia_)

    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs_k, marker='+',label="k-means++")
    ax.plot(nums,ARIs_r, marker='+',label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances_r, marker='*',label="k-means++")
    ax.plot(nums,Distances_k, marker='*',label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")
    ax.legend(loc="best")
    fig.suptitle("KMeans")
    plt.show()


# X, lable_true = create__datas([[2,2]], 1000, 0.5)
X, lable_true = create__datas([[1,1],[2,2],[1,2],[3,4]], 1000, 0.5)
test_KMeans(X, lable_true)
test_KMeans_nclusters(X, lable_true)
test_KMeans_n_init(X, lable_true)

# X_train,X_test, Y_train,Y_test = create_regression_digits(1000)
# test_KNeighborsRegressor(X_train,X_test, Y_train,Y_test)
# test_KNeighborsRegressor_k_w(X_train,X_test, Y_train,Y_test)
# test_KNeighborsRegressor_k_p(X_train,X_test, Y_train,Y_test)
