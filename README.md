# ASTRA-TAD
## Real-Time Telemetry Anomaly Detection System

ASTRA-TAD is a real-time telemetry anomaly detection system designed
to identify abnormal behavior in streaming sensor data.

The system combines Kafka-based telemetry streaming, a PyTorch
autoencoder, anomaly scoring, automated alert generation, and a
FastAPI monitoring dashboard.

---

# 1. Project Objective

The main objective of ASTRA-TAD is to:

- Collect telemetry data in real time
- Process streaming sensor measurements
- Detect abnormal telemetry using machine learning
- Calculate anomaly scores
- Generate severity-based security/telemetry alerts
- Identify possible contributing sensor features
- Display alerts through a web dashboard
- Evaluate model performance using ground-truth data

---

# 2. System Architecture

Telemetry Generator
        |
        v
Apache Kafka
        |
        v
Telemetry Topic
        |
        v
ML Consumer
        |
        v
Data Normalization
        |
        v
PyTorch Autoencoder
        |
        v
Reconstruction Error
        |
        v
Anomaly Threshold
        |
        +----------------+
        |                |
     Normal           Anomaly
                         |
                         v
                   Alert Logger
                         |
                         v
                   alerts.jsonl
                         |
                         v
                    FastAPI
                         |
                         v
                    Dashboard

---

# 3. Technologies Used

## Programming

- Python

## Machine Learning

- PyTorch
- Autoencoder
- Reconstruction-error based anomaly detection

## Streaming

- Apache Kafka
- Kafka Producer
- Kafka Consumer

## Backend

- FastAPI
- Uvicorn

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Frontend

- HTML
- CSS
- JavaScript

---

# 4. Telemetry Features

ASTRA-TAD processes seven telemetry features:

1. Temperature
2. Voltage
3. Current
4. Pressure
5. Vibration
6. RPM
7. Battery

---

# 5. Machine Learning Model

ASTRA-TAD uses an Autoencoder.

Architecture:

7
|
v
16
|
v
8
|
v
3
|
v
8
|
v
16
|
v
7

The three-neuron bottleneck provides a compressed representation
of the telemetry data.

The model reconstructs the input telemetry.

The reconstruction error is used as the anomaly score.

---

# 6. Anomaly Detection

The system calculates Mean Squared Error (MSE):

MSE = mean((original - reconstructed)^2)

If:

anomaly_score > threshold

the telemetry record is classified as anomalous.

Current threshold:

2.148575

---

# 7. Anomaly Types

The synthetic dataset contains several anomaly categories:

- Temperature spike
- Voltage drop
- Sensor drift
- Stuck sensor
- High vibration
- Correlated failure
- Battery degradation

---

# 8. Real-Time Processing

The telemetry generator publishes sensor data to the Kafka topic:

telemetry

The ML consumer reads the telemetry stream and performs:

1. Feature extraction
2. Normalization
3. Autoencoder inference
4. Reconstruction-error calculation
5. Threshold comparison
6. Root-cause feature estimation
7. Alert generation

---

# 9. Alert Severity

Alerts are classified using anomaly score thresholds.

CRITICAL:
score >= threshold × 10

HIGH:
score >= threshold × 5

MEDIUM:
score >= threshold × 2

LOW:
score >= threshold

---

# 10. Alert Information

Each generated alert contains information such as:

- Alert ID
- Timestamp
- Alert type
- Severity
- Sensor information
- Anomaly score
- Threshold
- Telemetry values
- Possible contributing features

---

# 11. API Endpoints

FastAPI runs on:

http://127.0.0.1:8001

Available endpoints:

GET /health

GET /alerts

GET /alerts/latest

GET /alerts/{alert_id}

Dashboard:

GET /

API documentation:

GET /docs

---

# 12. Dashboard

The dashboard provides:

- API status
- Total alerts
- Critical alerts
- High alerts
- Medium alerts
- Severity distribution
- Latest telemetry
- Anomaly score visualization
- Threshold visualization
- Recent alerts
- Severity filtering
- Automatic refresh

---

# 13. Evaluation

The project evaluates the model using ground-truth anomaly labels.

Evaluation includes:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1 Score
- False Positive Rate
- False Negative Rate
- Anomaly-type detection rate
- Anomaly score statistics

---

# 14. Evaluation Visualizations

Generated evaluation files include:

evaluation/anomaly_score_distribution.png

evaluation/anomaly_score_stream.png

evaluation/detection_rate_by_type.png

evaluation/mean_score_by_type.png

evaluation/evaluation_scores.csv

---

# 15. Project Structure

ASTRA-TAD/
|
├── api/
│   └── main.py
|
├── alerts/
│   └── alerts.jsonl
|
├── dashboard/
│   └── index.html
|
├── evaluation/
│   ├── evaluate_model.py
│   ├── visualize_results.py
│   ├── anomaly_score_distribution.png
│   ├── anomaly_score_stream.png
│   ├── detection_rate_by_type.png
│   ├── mean_score_by_type.png
│   └── evaluation_scores.csv
|
├── models/
│   ├── telemetry_autoencoder.pth
│   └── telemetry_scaler.joblib
|
├── simulator/
│   ├── anomaly_generator.py
│   └── telemetry_generator.py
|
├── streaming/
│   ├── __init__.py
│   ├── ml_consumer.py
│   └── alert_logger.py
|
├── .venv/
|
└── README.md

---

# 16. How to Run

Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Start Kafka.

Start the ML consumer:

python -m streaming.ml_consumer

Start FastAPI:

python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8001

Open the dashboard:

http://127.0.0.1:8001/

---

# 17. Generate Anomalous Telemetry

Run:

python simulator\anomaly_generator.py

This generates anomalous telemetry and publishes it to Kafka.

---

# 18. Evaluate the Model

Run:

python evaluation\evaluate_model.py

---

# 19. Generate Visualizations

Run:

python evaluation\visualize_results.py

---

# 20. Future Improvements

Possible future improvements include:

- Adaptive anomaly thresholds
- More advanced deep-learning architectures
- LSTM-based temporal anomaly detection
- Online model retraining
- Persistent database storage
- Authentication and authorization
- Email/SMS notification
- Grafana integration
- Docker deployment
- Kubernetes deployment
- Cloud deployment
- Advanced root-cause analysis

---

# 21. Conclusion

ASTRA-TAD demonstrates an end-to-end real-time anomaly detection
pipeline.

The system combines streaming telemetry, machine learning,
automated alert generation, API services, and visualization into
a single monitoring platform.

It can be extended for industrial telemetry, spacecraft/satellite
monitoring, IoT systems, infrastructure monitoring, and other
real-time sensor applications.