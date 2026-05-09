# US Flight Data Pipeline & MLOps Capstone

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository processes large-scale aviation data from the **US Bureau of Transportation Statistics**. It demonstrates an end-to-end data lifecycle, from ingesting raw batch and streaming data to training and serving a predictive machine learning model for flight delays.

**Dataset:** [Data Expo 2009: Airline on-time data (2008.csv)](https://www.kaggle.com/datasets/wenxingdi/data-expo-2009-airline-on-time-data/data?select=2008.csv)

### Key Features
* **Local WSL Environment:** Full Linux execution environment running natively on Windows.
* **Data Engineering:** PySpark for heavy-duty big data aggregations and Parquet file optimization.
* **Containerization:** Fully containerized Apache Kafka environments using Docker & Docker Compose.
* **MLOps Integration:** Model tracking, metric logging, and registry using MLflow (Local Mode).
* **Business Intelligence:** Automated visualization generation using Matplotlib.

---

## Architecture

The pipeline is split into three core components:

1.  **Batch Processing Pipeline (`/src/batch`):**
    * Ingests the historical 2008 flight dataset (CSV to Parquet).
    * Cleans, transforms, and aggregates data using PySpark to extract historical delay patterns.
    * Generates business intelligence visualizations.
2.  **MLOps Pipeline (`/src/mlops`):**
    * Consumes the processed batch features to train an XGBoost predictive model for flight delays.
    * Tracks experiments, hyperparameters, and model artifacts locally via MLflow.
3.  **Streaming Pipeline (`/src/streaming`):**
    * Simulates real-time flight telemetry using Apache Kafka.
    * Utilizes a Producer/Consumer architecture for high-throughput live data processing.

---

## Stack

* **Environment:** WSL2 (Ubuntu), Docker Desktop
* **Language:** Python 3.10+
* **Data Processing:** Pandas, PySpark, Java 11 (OpenJDK)
* **Machine Learning:** Scikit-Learn, XGBoost
* **MLOps:** MLflow
* **Streaming:** Apache Kafka (`kafka-python`)
* **Visualizations:** Matplotlib

---

## Directory Structure

```text
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
│   │   └───visuals             # Auto-generated BI charts (.png)
│   ├───processed               # Cleaned Parquet files
│   └───raw                     # Place 2008.csv here
├───infrastructure
│       serverless.yml
├───mlruns                      # Auto-generated MLflow tracking data
├───src
│   │   __init__.py
│   │
│   ├───batch
│   │       analytics.py        # PySpark aggregations
│   │       clean_data.py       # CSV to Parquet conversion
│   │       get_zipfiles.py     # AWS S3 ingestion (Cloud mode)
│   │       visualize.py        # Generates business dashboards
│   │       __init__.py
│   │
│   ├───mlops
│   │       train_model.py      # XGBoost training & MLflow logging
│   │       __init__.py
│   │
│   └───streaming
│           consumer.py         # Kafka listener
│           producer.py         # Kafka data sender
│           __init__.py
│
└───tests
        test_clean_data.py
        __init__.py

```

---

## Setup & Execution Guide

### Phase 1: Environment Setup (WSL & Python)

This project is designed to run in a Linux environment using WSL (Windows Subsystem for Linux).

1. Open your Ubuntu WSL terminal and navigate to the project folder.
2. Install the required Java engine for PySpark:
```bash
sudo apt update
sudo apt install openjdk-11-jre-headless -y
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

```


3. Create and activate a Python Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate

```


4. Install dependencies:
```bash
pip install -r requirements.txt

```


*(Ensure pandas, pyspark, mlflow, xgboost, scikit-learn, matplotlib, and kafka-python are installed).*

### Phase 2: Data Acquisition

1. Download the `2008.csv` file from the Kaggle link above.
2. Place it exactly at: `data/raw/2008.csv`.

### Phase 3: The Batch Data Engineering Pipeline

Ensure your virtual environment is active (`source venv/bin/activate`) before running these scripts.

1. **Clean the Data (CSV -> Parquet):**
```bash
python src/batch/clean_data.py

```


2. **Run PySpark Analytics:**
```bash
python src/batch/analytics.py

```


*This answers the Capstone analytics questions and generates `ml_features.parquet`.*
3. **Generate Visualizations:**
```bash
python src/batch/visualize.py

```


*Check `data/analytics/visuals/` for the outputted charts.*

### Phase 4: MLOps Pipeline (Model Training)

Train the XGBoost model using the features generated in Phase 3.

1. **Train the Model:**
```bash
python src/mlops/train_model.py

```


2. **View the MLflow Dashboard:**
MLflow saves all metrics and model artifacts locally to the `mlruns/` folder. To view the UI, start the server:
```bash
mlflow ui --port 2000

```


Open your Windows web browser and navigate to: **http://localhost:2000**

### Phase 5: Real-Time Streaming Pipeline (Kafka)

To simulate live flight telemetry, we use Apache Kafka running in Docker.

**1. Start the Kafka Infrastructure:**

```bash
docker compose up -d

```

*Wait 15-30 seconds for the Kafka broker to fully boot.*

**2. Start the Consumer (Terminal 1):**
In your current WSL terminal, activate the environment and start listening for data:

```bash
source venv/bin/activate
python src/streaming/consumer.py

```

**3. Start the Producer (Terminal 2):**
Open a **new** WSL terminal window, activate the environment, and start streaming the CSV data into Kafka:

```bash
cd ~/Cloud-Computing-Project
source venv/bin/activate
python src/streaming/producer.py

```

*You will immediately see the Producer sending messages in Terminal 2, and the Consumer processing them in Terminal 1.*

```
