import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(f):
    data = pd.read_csv(f, header=None)

    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy() / 255

    return X, y


def assign_clusters(points, centroids):
    distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_means(X, labels, k):
    new_means = np.array([
        X[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(X.shape[1])
        for i in range(k)])
    return new_means


def k_means(X, k, v=True, end_condition=1e-5, max_iter=1000):
    means = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iter):
        labels = assign_clusters(X, means)
        new_means = update_means(X, labels, k)

        change = np.linalg.norm(new_means - means)
        if(v and not i % 5):
            print(f'{i}. Change: {change}')
        if change <= end_condition:
            break
        means = new_means

    return means, labels


def evaluate_cluster_homogeneity(y, labels, k):
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

        print(f"Cluster {i}: Most common = {most_common_label}, "
              f"Homogeneity = {homogeneity:.2f}, Size = {len(cluster_indices)}")

    average_homogeneity = np.mean(homogeneity_scores)
    print(f"\nAverage cluster homogeneity: {average_homogeneity:.2f}")


np.random.seed(40)
X, y = load_dataset('mnist.csv')
k = 10
centroids, labels = k_means(X, k, v=True)

print()
evaluate_cluster_homogeneity(y, labels, k)

fig, axs = plt.subplots(1, k, figsize=(15, 2))
for i, ax in enumerate(axs):
    ax.imshow(centroids[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle("Cluster Means for MNIST")
plt.show()
