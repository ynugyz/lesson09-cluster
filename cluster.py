"""第 9 课银行客户分群。读取课程原始文件，在 results/ 生成结果。"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "bank_customers.csv"
RESULTS = ROOT / "results"
SOURCE_SHA256 = "b82ef3124080469b7f208e246f08b6de66bc2a057e0aa84d6eea481716d578da"
FEATURES = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
EXTENDED = FEATURES + ["account_age_months"]
COLORS = ["#0072B2", "#D55E00", "#8C519F", "#00866A"]
BASELINE_SEED = 42
K = 4


def fit(data, k=K, seed=BASELINE_SEED):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)


def profile(df, labels, features):
    groups = df.assign(cluster=labels).groupby("cluster", sort=True)
    return groups[features].mean().assign(customers=groups.size())


def records(table):
    return table.reset_index().round(6).to_dict(orient="records")


def save_figure(fig, filename):
    fig.savefig(RESULTS / filename, dpi=170, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def plot_elbow(sse, elbow):
    ks = np.arange(1, len(sse) + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.3), layout="constrained")
    ax.plot(ks, np.linspace(sse[0], sse[-1], len(sse)), "--", color="#9CA7AF",
            linewidth=1, label="Line between endpoints")
    ax.plot(ks, sse, "o-", color=COLORS[0], linewidth=2, label="Observed SSE")
    ax.scatter(elbow, sse[elbow - 1], s=95, color=COLORS[1], zorder=4)
    ax.annotate(f"Elbow candidate: k={elbow}", (elbow, sse[elbow - 1]),
                xytext=(-25, -35), textcoords="offset points", ha="right",
                arrowprops={"arrowstyle": "->", "color": "#53636E"})
    ax.set(title="Elbow curve | Bank customers", xlabel="Number of clusters (k)",
           ylabel="Within-cluster sum of squares (SSE)", xticks=ks)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "elbow.png")


def plot_profiles(centers):
    fig, ax = plt.subplots(figsize=(8.0, 4.4), layout="constrained")
    x = np.arange(len(FEATURES))
    width = 0.18
    for i, (center, color) in enumerate(zip(centers, COLORS)):
        ax.bar(x + (i - 1.5) * width, center, width, color=color, label=f"Cluster {i}")
    ax.axhline(0, color="#576570", linewidth=0.8)
    ax.set(title="Four customer profiles | Standardized means",
           ylabel="Mean in standard-deviation units",
           xticks=x, xticklabels=["Balance", "Monthly spend", "Products", "Overdues"])
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    save_figure(fig, "profiles.png")


def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(7.6, 4.7), layout="constrained")
    for (cluster, group), color in zip(df.groupby("cluster", sort=True), COLORS):
        ax.scatter(group["balance"] / 1000, group["avg_monthly_spend"] / 1000,
                   color=color, s=26, alpha=0.80, edgecolor="white", linewidth=0.3,
                   label=f"Cluster {cluster} ({len(group)})")
    ax.set(title="Cluster assignments | Balance and monthly spend",
           xlabel="Balance (thousand yuan)", ylabel="Monthly spend (thousand yuan)")
    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save_figure(fig, "scatter.png")


def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("bank_customers.csv 的 SHA-256 与课程原始文件不一致")
    df = pd.read_csv(SOURCE)
    RESULTS.mkdir(exist_ok=True)

    # 基线分群仅使用四个行为特征。原始表和标签字段留在模型之外。
    scaled = StandardScaler().fit_transform(df[FEATURES])
    models = [fit(scaled, k) for k in range(1, 10)]
    baseline = models[K - 1]
    df["cluster"] = baseline.labels_
    base = profile(df, baseline.labels_, FEATURES)
    df[["customer_id", "cluster"]].to_csv(RESULTS / "assignments.csv", index=False)

    sse = np.array([model.inertia_ for model in models])
    # 各点到端点连线的垂距，与其纵向差值成正比。
    elbow = int(np.argmax(np.linspace(sse[0], sse[-1], len(sse)) - sse) + 1)
    scores = {k: round(float(silhouette_score(scaled, model.labels_)), 6)
              for k, model in enumerate(models[1:], start=2)}

    comparison = {}
    for seed in (42, 0, 7):
        model = baseline if seed == BASELINE_SEED else fit(scaled, seed=seed)
        comparison[seed] = {
            "sse": round(float(model.inertia_), 6),
            "ari_vs_seed42": round(float(adjusted_rand_score(baseline.labels_, model.labels_)), 6),
            "profiles": records(profile(df, model.labels_, FEATURES)),
        }

    extended_labels = fit(StandardScaler().fit_transform(df[EXTENDED])).labels_
    raw_labels = fit(df[FEATURES]).labels_
    cross = pd.crosstab(df["cluster"], extended_labels)
    raw_variance = df[FEATURES].var(ddof=0)
    original_profile = profile(df, baseline.labels_, EXTENDED)
    results = {
        "customers": len(df),
        "input_sha256": digest,
        "features": FEATURES,
        "missing_in_used_features": df[EXTENDED].isna().sum().to_dict(),
        "baseline_profiles": records(original_profile),
        "sse": dict(enumerate(sse.round(6).tolist(), start=1)),
        "elbow_k": elbow,
        "silhouette": scores,
        "seed_comparison": comparison,
        "extended_profiles": records(profile(df, extended_labels, EXTENDED)),
        "extended_ari_vs_baseline": round(float(adjusted_rand_score(baseline.labels_, extended_labels)), 6),
        "baseline_to_extended_counts": cross.to_numpy().tolist(),
        "unscaled_profiles": records(profile(df, raw_labels, FEATURES).sort_values("balance")),
        "unscaled_ari_vs_baseline": round(float(adjusted_rand_score(baseline.labels_, raw_labels)), 6),
        "unscaled_balance_variance_share": round(float(raw_variance["balance"] / raw_variance.sum()), 6),
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_elbow(sse, elbow)
    plot_profiles(baseline.cluster_centers_)
    plot_scatter(df)
    print(base.round(2).to_string())
    print(f"\nCustomers: {len(df)} | Elbow: k={elbow} | Silhouette (k=4): {scores[K]:.4f}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
