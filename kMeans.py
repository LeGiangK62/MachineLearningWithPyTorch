from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


# kmeans_display(X, original_label)

def k_means(nodeLocation, numberOfClusters):

    # Initialize the cluster centers
    cluster_centers = k_means_center(nodeLocation, numberOfClusters)

    # Initialize the labels for each point
    labels = []

    while True:
        # Assign each point to the closest cluster center
        labels .append(k_means_assign_cluster(nodeLocation, numberOfClusters))
        # If the point is already assigned to a cluster center, end the algorithm
        if k_means_check(labels[-1], labels[-2]):
            break
        # Update the cluster centers
        cluster_centers = k_means_update_center(nodeLocation, numberOfClusters)

    return (cluster_centers, labels)


def k_means_center(nodeLocation, numberOfClusters):
    return nodeLocation[np.random.choice(X.shape[0], numberOfClusters, replace=False)]


def k_means_assign_cluster(nodeLocation, centerLocation):
    distance = cdist(nodeLocation, centerLocation)
    return np.argmin(distance, axis = 1)


def k_means_update_center(node_location, cluster_list, number_of_clusters):
    centers = np.zeros((number_of_clusters, node_location.shape[1]))
    for k in range(number_of_clusters):
        # collect all points assigned to the k-th cluster
        cluster_location = node_location[labels == number_of_clusters, :]
        # take average
        centers[k, :] = np.mean(cluster_location, axis=0)
    return centers

def k_means_check(lableNew, lableOld):
    return ()


np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

print(original_label)

(cluster_centers, labels) = k_means(X, K)


