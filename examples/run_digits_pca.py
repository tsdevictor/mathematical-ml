import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from mathematical_ml.decomposition import PCA


def main():
    digits = load_digits()
    X = digits.data
    y = digits.target

    Z = PCA(n_components=2).fit_transform(X)

    plt.figure()
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8)
    plt.title("PCA projection of handwritten digits")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="digit")
    plt.savefig("figures/digits_pca.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
