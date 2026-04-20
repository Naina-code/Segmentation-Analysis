import mysql.connector
import pandas as pd
import streamlit as st

# ── update these or use st.secrets ──────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "your_password",   # ← change this
    "database": "customer_segmentation"
}
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def save_session(filename: str, num_rows: int, num_clusters: int) -> int:
    """Insert a new upload session and return its ID."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO upload_sessions (filename, num_rows, num_clusters) VALUES (%s, %s, %s)",
        (filename, num_rows, num_clusters)
    )
    conn.commit()
    session_id = cur.lastrowid
    cur.close()
    conn.close()
    return session_id


def save_customers(session_id: int, df: pd.DataFrame):
    """Bulk-insert all customer rows with their cluster labels."""
    conn = get_connection()
    cur  = conn.cursor()
    rows = [
        (
            session_id,
            int(row["Gender"]),
            int(row["Age"]),
            float(row["Annual Income (k$)"]),
            float(row["Spending Score (1-100)"]),
            int(row["Cluster"])
        )
        for _, row in df.iterrows()
    ]
    cur.executemany(
        """INSERT INTO customer_clusters
           (session_id, gender, age, annual_income, spending_score, cluster_label)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        rows
    )
    conn.commit()
    cur.close()
    conn.close()


def save_profiles(session_id: int, profiles: pd.DataFrame):
    """Insert cluster-level summary stats."""
    conn = get_connection()
    cur  = conn.cursor()
    for _, row in profiles.iterrows():
        cur.execute(
            """INSERT INTO cluster_profiles
               (session_id, cluster_label, avg_age, avg_income, avg_spending, customer_count)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                session_id,
                int(row["Cluster"]),
                float(row["Age"]),
                float(row["Annual Income (k$)"]),
                float(row["Spending Score (1-100)"]),
                int(row["count"])
            )
        )
    conn.commit()
    cur.close()
    conn.close()


def fetch_sessions() -> pd.DataFrame:
    """Return all past upload sessions."""
    conn = get_connection()
    df   = pd.read_sql("SELECT * FROM upload_sessions ORDER BY uploaded_at DESC", conn)
    conn.close()
    return df


def fetch_session_customers(session_id: int) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql(
        "SELECT * FROM customer_clusters WHERE session_id = %s",
        conn, params=(session_id,)
    )
    conn.close()
    return df
