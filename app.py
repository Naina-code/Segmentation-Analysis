import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np

from model import preprocess, find_best_k, run_kmeans, cluster_profiles, label_cluster, FEATURE_COLS
from db    import save_session, save_customers, save_profiles, fetch_sessions

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding: 2rem 3rem; }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: #0f172a;
        margin-bottom: 0;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        font-family: 'Space Mono', monospace;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .cluster-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🛍️ Customer Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your customer CSV — get instant AI-powered cluster analysis saved to MySQL</p>', unsafe_allow_html=True)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    auto_k     = st.toggle("Auto-detect best K", value=True)
    manual_k   = st.slider("Manual K (clusters)", 2, 10, 5, disabled=auto_k)
    save_to_db = st.toggle("💾 Save results to MySQL", value=True)

    st.divider()
    st.markdown("### 📋 Required CSV Columns")
    st.code("Gender\nAge\nAnnual Income (k$)\nSpending Score (1-100)", language="text")

    st.divider()
    st.markdown("### 🗂 Past Sessions")
    try:
        sessions = fetch_sessions()
        if sessions.empty:
            st.info("No sessions yet.")
        else:
            st.dataframe(sessions[["id","filename","num_clusters","uploaded_at"]], use_container_width=True)
    except Exception as e:
        st.warning(f"DB not connected: {e}")


# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Mall_Customers.csv or any compatible CSV", type=["csv"])

if uploaded is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()


# ── Load & Preview ────────────────────────────────────────────────────────────
raw_df = pd.read_csv(uploaded)
st.markdown('<div class="section-header">📄 Raw Data Preview</div>', unsafe_allow_html=True)
st.dataframe(raw_df.head(10), use_container_width=True)

# Validate columns
required = {"Age", "Annual Income (k$)", "Spending Score (1-100)"}
if not required.issubset(set(raw_df.columns)):
    st.error(f"❌ Missing columns: {required - set(raw_df.columns)}")
    st.stop()


# ── Preprocess ────────────────────────────────────────────────────────────────
df = preprocess(raw_df)


# ── Find best K ───────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(df[FEATURE_COLS])

st.markdown('<div class="section-header">📊 Optimal Cluster Analysis</div>', unsafe_allow_html=True)
results = find_best_k(X_scaled)
best_k  = results["best_k"] if auto_k else manual_k

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(results["k_range"], results["wcss"], "o-", color="#3b82f6", linewidth=2)
    ax.set_xlabel("Number of Clusters"); ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method", fontweight="bold")
    ax.grid(alpha=0.3); fig.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    colors  = ["#ef4444" if k == best_k else "#94a3b8" for k in results["k_range"]]
    ax.bar(results["k_range"], results["silhouettes"], color=colors)
    ax.set_xlabel("Number of Clusters"); ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Scores (red = best)", fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    st.pyplot(fig)

st.success(f"✅ Best K = **{best_k}** {'(auto-detected)' if auto_k else '(manual)'}")


# ── Run Clustering ────────────────────────────────────────────────────────────
df_clustered = run_kmeans(df, best_k)
profiles     = cluster_profiles(df_clustered)
df_clustered["Segment"] = df_clustered.apply(label_cluster, axis=1)


# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Results Summary</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

metrics = [
    (m1, len(df_clustered),                     "Total Customers"),
    (m2, best_k,                                "Clusters Found"),
    (m3, int(df_clustered["Age"].mean()),        "Avg Age"),
    (m4, int(df_clustered["Annual Income (k$)"].mean()), "Avg Income (k$)"),
]
for col, val, label in metrics:
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{val}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)


# ── Cluster Scatter Plot ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺 Cluster Visualization</div>', unsafe_allow_html=True)

palette = sns.color_palette("Set1", best_k)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Income vs Spending
for cluster_id in range(best_k):
    subset = df_clustered[df_clustered["Cluster"] == cluster_id]
    axes[0].scatter(
        subset["Annual Income (k$)"], subset["Spending Score (1-100)"],
        label=f"Cluster {cluster_id}", color=palette[cluster_id], alpha=0.7, s=60
    )
axes[0].set_xlabel("Annual Income (k$)"); axes[0].set_ylabel("Spending Score")
axes[0].set_title("Income vs Spending Score", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Age vs Spending
for cluster_id in range(best_k):
    subset = df_clustered[df_clustered["Cluster"] == cluster_id]
    axes[1].scatter(
        subset["Age"], subset["Spending Score (1-100)"],
        label=f"Cluster {cluster_id}", color=palette[cluster_id], alpha=0.7, s=60
    )
axes[1].set_xlabel("Age"); axes[1].set_ylabel("Spending Score")
axes[1].set_title("Age vs Spending Score", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)


# ── Cluster Profiles ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧬 Cluster Profiles</div>', unsafe_allow_html=True)

display_profiles = profiles.copy()
display_profiles.columns = ["Cluster", "Avg Age", "Avg Income (k$)", "Avg Spending Score", "Count"]
st.dataframe(display_profiles, use_container_width=True)

# Segment distribution
seg_counts = df_clustered["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]

fig, ax = plt.subplots(figsize=(7, 4))
colors  = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"]
ax.barh(seg_counts["Segment"], seg_counts["Count"], color=colors[:len(seg_counts)])
ax.set_xlabel("Number of Customers")
ax.set_title("Customer Segment Distribution", fontweight="bold")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
st.pyplot(fig)


# ── Full Clustered Table ──────────────────────────────────────────────────────
with st.expander("📋 View Full Clustered Dataset"):
    st.dataframe(df_clustered, use_container_width=True)

    csv = df_clustered.to_csv(index=False).encode()
    st.download_button("⬇️ Download Clustered CSV", csv, "clustered_customers.csv", "text/csv")


# ── Save to MySQL ─────────────────────────────────────────────────────────────
if save_to_db:
    st.markdown('<div class="section-header">💾 Save to MySQL</div>', unsafe_allow_html=True)
    if st.button("Save Results to Database", type="primary"):
        try:
            with st.spinner("Saving to MySQL..."):
                session_id = save_session(uploaded.name, len(df_clustered), best_k)
                save_customers(session_id, df_clustered)
                save_profiles(session_id, profiles)
            st.success(f"✅ Saved to MySQL! Session ID: `{session_id}`")
        except Exception as e:
            st.error(f"❌ DB Error: {e}")
            st.info("Make sure MySQL is running and DB_CONFIG in db.py is correct.")
