# ASTRA-TAD — AI/ML Models & Detection Algorithms

## 1. Overview

ASTRA-TAD is a real-time telemetry anomaly detection system that uses
machine learning techniques to identify abnormal behavior in telemetry data.

The project contains multiple machine learning and detection approaches:

- Isolation Forest
- Dense Autoencoder
- LSTM Autoencoder
- Hybrid Detection
- Score-Based Fusion
- Root-Cause Feature Analysis

The real-time pipeline currently uses the trained telemetry Autoencoder
for streaming anomaly detection.

---

# 2. Telemetry Input

ASTRA-TAD works with seven telemetry features:

| Feature | Description |
|---|---|
| Temperature | Thermal condition of the system |
| Voltage | Electrical voltage |
| Current | Electrical current |
| Pressure | System pressure |
| Vibration | Mechanical vibration |
| RPM | Rotational speed |
| Battery | Battery level |

These features are transformed into numerical feature vectors before being
passed to the machine learning models.

---

# 3. Data Preprocessing

Machine learning models require telemetry features to be represented in a
consistent numerical scale.

The preprocessing pipeline includes:

```text
Raw Telemetry
      ↓
Feature Extraction
      ↓
Numerical Conversion
      ↓
Feature Scaling
      ↓
ML Model