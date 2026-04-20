-- Run this file once to set up your MySQL database

CREATE DATABASE IF NOT EXISTS customer_segmentation;
USE customer_segmentation;

-- Stores each uploaded file session
CREATE TABLE IF NOT EXISTS upload_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_rows INT,
    num_clusters INT
);

-- Stores each customer row with its assigned cluster
CREATE TABLE IF NOT EXISTS customer_clusters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    gender INT,
    age INT,
    annual_income FLOAT,
    spending_score FLOAT,
    cluster_label INT,
    FOREIGN KEY (session_id) REFERENCES upload_sessions(id) ON DELETE CASCADE
);

-- Stores cluster summary stats per session
CREATE TABLE IF NOT EXISTS cluster_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    cluster_label INT,
    avg_age FLOAT,
    avg_income FLOAT,
    avg_spending FLOAT,
    customer_count INT,
    FOREIGN KEY (session_id) REFERENCES upload_sessions(id) ON DELETE CASCADE
);
