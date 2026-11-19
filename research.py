"""Compare centroid and density clustering on two simulated concentric rings."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score


def main():
    points, ring_labels = make_circles(
        n_samples=600, factor=0.5, noise=0.035, random_state=42
    )
    kmeans_labels = KMeans(
        n_clusters=2, n_init=10, random_state=42
    ).fit_predict(points)
    dbscan_labels = DBSCAN(eps=0.17, min_samples=5).fit_predict(points)

    fig, axes = plt.subplots(
        1, 3, figsize=(10.6, 3.9), sharex=True, sharey=True, layout="constrained"
    )
    fig.suptitle("Clustering concentric rings", fontsize=16, fontweight="bold")
    colors = ["#1864A0", "#BC4C21"]
    for ax, title, labels in zip(
        axes,
        ("Synthetic ring labels", "KMeans (k = 2)", "DBSCAN (eps = 0.17)"),
        (ring_labels, kmeans_labels, dbscan_labels),
    ):
        for label in sorted(set(labels)):
            group = points[labels == label]
            color = "#777777" if label == -1 else colors[label % len(colors)]
            ax.scatter(group[:, 0], group[:, 1], s=12, color=color, alpha=0.85)
        ax.set(
            title=title, xlabel="x", aspect="equal",
            xlim=(-1.2, 1.2), ylim=(-1.2, 1.2),
        )
        ax.grid(color="#e5e7eb", linewidth=0.6)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("y")
            continue
        ari = adjusted_rand_score(ring_labels, labels)
        cluster_count = len(set(labels) - {-1})
        noise_count = int((labels == -1).sum())
        ax.set_title(
            f"{title}\nARI = {ari:.3f}  |  {cluster_count} clusters, {noise_count} noise"
        )
        print(f"{title}: ARI={ari:.6f}, clusters={cluster_count}, noise={noise_count}")

    output = Path(__file__).resolve().parent / "results" / "rings.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
