import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist


def load_data(f, n_samples=12000):
    data = pd.read_csv(f)
    X = data.iloc[:n_samples, 1:].to_numpy() / 255
    y = data.iloc[:n_samples, 0].to_numpy()
    return X, y


def make_knn(X, k=30):
    A = kneighbors_graph(X, n_neighbors=k)
    A = A + A.T - A * A.T
    return A


def make_L(A):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L


def spectral_clustering(X, l=3, k=10):
    A = make_knn(X)
    L = make_L(A)

    evals, evecs = eigsh(L, k=l)
    evecs = evecs[:, 1:]  # skip trivial constant eigenvector

    X_embed = evecs[:, :l]
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_embed)
    centroids = kmeans.cluster_centers_

    return X_embed, clusters, centroids


def evaluate_homogeneity(y, labels, k):
    homogeneity_scores = []

    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        true_labels = y[cluster_indices]

        if len(true_labels) == 0:
            homogeneity_scores.append(0)
            continue

        counts = np.bincount(true_labels, minlength=10)
        most_common_label = np.argmax(counts)
        max_count = counts[most_common_label]

        homogeneity = max_count / len(true_labels)
        homogeneity_scores.append(homogeneity)

    return np.mean(homogeneity_scores)


def show_imgs(f, closest_indices):
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(closest_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx].reshape((28, 28)), cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.savefig(f)
    plt.show()


def test_l(X, y, k, l_max=25):
    avg_h_scores = []
    for l in range(2, l_max):
        X_embed, clusters, centroids = spectral_clustering(X, l, k)
        closest_indices = np.argmin(cdist(centroids, X_embed), axis=1)
        show_imgs(f'imgs/l{l}.png', closest_indices)
        avg_h_scores.append(evaluate_homogeneity(y, clusters, k))
        print(f'Avg. cluster homogeneity (l={l}): {avg_h_scores[-1]:.2f}')
    return avg_h_scores


k = 10
X, y = load_data('mnist.csv', 12000)
avg_h = test_l(X, y, k)
print(f'Best l: {avg_h.index(max(avg_h)) + 2}')
