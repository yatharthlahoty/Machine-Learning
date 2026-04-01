# ============================================================
# Question 2
# Customer Segmentation using K-Means Clustering
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# ── 1. Generate synthetic customer dataset ─────────────────
np.random.seed(42)
n = 300

age              = np.random.randint(18, 70, n)
annual_income    = np.random.randint(15000, 120000, n)
spending_score   = np.random.randint(1, 100, n)
purchase_freq    = np.random.randint(1, 50, n)
avg_order_value  = np.random.uniform(20, 500, n)

df = pd.DataFrame({
    "Age":             age,
    "Annual_Income":   annual_income,
    "Spending_Score":  spending_score,
    "Purchase_Freq":   purchase_freq,
    "Avg_Order_Value": avg_order_value,
})

print("=" * 55)
print("  CUSTOMER SEGMENTATION - K-Means Clustering")
print("=" * 55)
print("\n── Dataset Overview ──")
print(df.describe().round(2))

# ── 2. Scale features ──────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ── 3. Find optimal K using Elbow + Silhouette ─────────────
k_range    = range(2, 11)
inertias   = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# ── 4. Train final model with optimal K ────────────────────
optimal_k = k_range[np.argmax(silhouettes)]
print(f"\n── Optimal K (highest Silhouette Score): {optimal_k} ──")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ── 5. Evaluate clustering quality ────────────────────────
sil  = silhouette_score(X_scaled, df["Cluster"])
db   = davies_bouldin_score(X_scaled, df["Cluster"])
ch   = calinski_harabasz_score(X_scaled, df["Cluster"])

print("\n── Clustering Evaluation Metrics ──")
print(f"  Silhouette Score        : {sil:.4f}  (higher is better, range: -1 to 1)")
print(f"  Davies-Bouldin Score    : {db:.4f}  (lower is better)")
print(f"  Calinski-Harabasz Score : {ch:.4f}  (higher is better)")
print(f"  Inertia (within-cluster): {kmeans.inertia_:.4f}")

print("\n── Cluster Summary (mean values) ──")
summary = df.groupby("Cluster").mean().round(2)
print(summary)

# ── 6. Visualizations ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Customer Segmentation - K-Means Clustering", fontsize=14)

colors = plt.cm.tab10.colors

# Plot 1 – Elbow Curve
axes[0, 0].plot(list(k_range), inertias, "bo-", markersize=6)
axes[0, 0].set_xlabel("Number of Clusters (K)")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].set_title("Elbow Method")
axes[0, 0].axvline(optimal_k, color="red", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0, 0].legend()

# Plot 2 – Silhouette Scores
axes[0, 1].plot(list(k_range), silhouettes, "rs-", markersize=6)
axes[0, 1].set_xlabel("Number of Clusters (K)")
axes[0, 1].set_ylabel("Silhouette Score")
axes[0, 1].set_title("Silhouette Score vs K")
axes[0, 1].axvline(optimal_k, color="blue", linestyle="--", label=f"Optimal K={optimal_k}")
axes[0, 1].legend()

# Plot 3 – PCA 2D scatter of clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
for cluster_id in range(optimal_k):
    mask = df["Cluster"] == cluster_id
    axes[1, 0].scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.7, s=40,
        color=colors[cluster_id % len(colors)]
    )
axes[1, 0].set_title("Clusters (PCA 2D View)")
axes[1, 0].set_xlabel("PCA Component 1")
axes[1, 0].set_ylabel("PCA Component 2")
axes[1, 0].legend()

# Plot 4 – Income vs Spending Score colored by cluster
for cluster_id in range(optimal_k):
    mask = df["Cluster"] == cluster_id
    axes[1, 1].scatter(
        df.loc[mask, "Annual_Income"],
        df.loc[mask, "Spending_Score"],
        label=f"Cluster {cluster_id}",
        alpha=0.7, s=40,
        color=colors[cluster_id % len(colors)]
    )
axes[1, 1].set_xlabel("Annual Income")
axes[1, 1].set_ylabel("Spending Score")
axes[1, 1].set_title("Income vs Spending Score by Cluster")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("customer_segmentation_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("=" * 55)
