import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FEATURE_COLS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the dataframe."""
    df = df.copy()

    # Drop ID column if present
    df = df.drop(columns=["CustomerID"], errors="ignore")

    # Encode gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
        df["Gender"] = df["Gender"].fillna(0)

    # Drop nulls in feature columns
    df = df.dropna(subset=FEATURE_COLS)

    return df


def find_best_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> dict:
    """Return WCSS and silhouette scores for each k."""
    wcss        = []
    silhouettes = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return {
        "k_range":     list(k_range),
        "wcss":        wcss,
        "silhouettes": silhouettes,
        "best_k":      list(k_range)[silhouettes.index(max(silhouettes))]
    }


def run_kmeans(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Scale features, run KMeans, return df with Cluster column."""
    X      = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km            = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df            = df.copy()
    df["Cluster"] = km.fit_predict(X_sc)

    return df


def cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-cluster mean stats + count."""
    profile = (
        df.groupby("Cluster")[FEATURE_COLS]
        .mean()
        .round(2)
        .reset_index()
    )
    counts          = df.groupby("Cluster").size().reset_index(name="count")
    profile         = profile.merge(counts, on="Cluster")
    return profile


CLUSTER_LABELS = {
    "High Income High Spenders":  "💎 Premium Targets",
    "High Income Low Spenders":   "💼 Conservative Wealthy",
    "Low Income High Spenders":   "🛍️ Impulse Buyers",
    "Low Income Low Spenders":    "💰 Budget Conscious",
    "Middle Income Middle Spend": "🎯 Average Customers",
}


def label_cluster(row: pd.Series) -> str:
    """Assign a human-readable label based on income & spending."""
    inc  = row["Annual Income (k$)"]
    spnd = row["Spending Score (1-100)"]
    if inc >= 70 and spnd >= 60:
        return "💎 Premium Targets"
    elif inc >= 70 and spnd < 40:
        return "💼 Conservative Wealthy"
    elif inc < 40 and spnd >= 60:
        return "🛍️ Impulse Buyers"
    elif inc < 40 and spnd < 40:
        return "💰 Budget Conscious"
    else:
        return "🎯 Average Customers"
