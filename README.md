# 🛍️ Customer Segmentation App

A Streamlit web app for customer segmentation using KMeans clustering, with MySQL storage.

## 📁 Project Structure

```
customer_segmentation/
├── app.py            ← Streamlit frontend (run this)
├── model.py          ← AI/ML clustering logic
├── db.py             ← MySQL connection & queries
├── schema.sql        ← Database setup script
├── requirements.txt  ← Python dependencies
└── README.md
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up MySQL database
Open MySQL and run:
```bash
mysql -u root -p < schema.sql
```
Or copy-paste the contents of `schema.sql` into MySQL Workbench.

### 3. Update DB credentials
Open `db.py` and update:
```python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "your_password",   # ← your MySQL password
    "database": "customer_segmentation"
}
```

### 4. Run the app
```bash
streamlit run app.py
```

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 📤 CSV Upload | Upload any customer CSV with Age, Income, Spending Score |
| 🤖 Auto K Detection | Automatically finds the best number of clusters using silhouette score |
| 📊 Visualizations | Elbow chart, silhouette scores, scatter plots |
| 🧬 Cluster Profiles | Mean stats per cluster with customer counts |
| 💾 MySQL Storage | Saves sessions, customer clusters, and profiles to DB |
| ⬇️ CSV Download | Download the clustered dataset |

## 📋 Required CSV Columns

Your CSV must contain:
- `Gender` (Male/Female)
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

Optional: `CustomerID` (auto-dropped if present)

## 🗄️ Database Tables

| Table | Description |
|-------|-------------|
| `upload_sessions` | Each file upload with timestamp |
| `customer_clusters` | All customers with cluster labels |
| `cluster_profiles` | Average stats per cluster |
