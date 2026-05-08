# US Flight Data Pipeline & MLOps Capstone

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

This repository processes large-scale aviation data from the **US Bureau of Transportation Statistics**. It demonstrates an end-to-end data lifecycle, from ingesting raw batch and streaming data to training and serving a predictive machine learning model for flight delays.
data = https://www.kaggle.com/datasets/wenxingdi/data-expo-2009-airline-on-time-data/data?select=2008.csv

### Key Difference
*   **Infrastructure as Code (IaC):** Automated provisioning using Terraform.
*   **Containerization:** Fully containerized environments using Docker & Docker Compose for reproducible local development.
*   **MLOps Integration:** Model tracking and registry using MLflow.
*   **CI/CD:** Automated testing and linting via GitHub Actions.
*   **Modern Python Standards:** Dependency management via `requirements.txt` (or Poetry) and strict linting (Black/Flake8).

---

## Architecture

The pipeline is split into three core components:

1.  **Batch Processing Pipeline (`/src/batch`):**
    *   Ingests historical flight dataset.
    *   Cleans, transforms, and aggregates data to extract historical delay patterns.
    *   Outputs processed features to cloud storage (Data Lake/Warehouse).
2.  **Streaming Pipeline (`/src/streaming`):**
    *   Simulates real-time flight telemetry and status updates.
    *   Processes high-throughput streams for real-time anomaly detection.
3.  **MLOps Pipeline (`/src/mlops`):**
    *   Consumes the processed batch features to train a predictive model (e.g., predicting the likelihood of a flight delay).
    *   Tracks experiments and model artifacts.

---

## Stack

*   **Language:** Python 3.9+
*   **Data Processing:** Pandas, PySpark (Optional, based on scale)
*   **Machine Learning:** Scikit-Learn, XGBoost
*   **MLOps:** MLflow
*   **Deployment & Ops:** Docker, Docker Compose, GitHub Actions, Terraform
*   **Cloud Provider:** [Insert Provider: AWS / Google Cloud]

---

## Dir

```
│   .gitignore
│   docker-compose.yml
│   Dockerfile
│   LICENSE.md
│   README.md
│   requirements.txt
│
├───.github
│   └───workflows
├───data
│   ├───analytics
│   ├───processed
│   └───raw
├───infrastructure
│       serverless.yml
│
├───src
│   │   __init__.py
│   │
│   ├───batch
│   │       analytics.py
│   │       clean_data.py
│   │       get_zipfiles.py
│   │       __init__.py
│   │
│   ├───mlops
│   │       train_model.py
│   │       __init__.py
│   │
│   └───streaming
│           consumer.py
│           producer.py
│           __init__.py
│
└───tests
        test_clean_data.py
        __init__.py
