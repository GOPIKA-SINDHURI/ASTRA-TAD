# ASTRA-TAD — End-to-End Demo Guide

This guide demonstrates the complete ASTRA-TAD real-time AI/ML anomaly detection pipeline.

---

## 1. System Overview

ASTRA-TAD processes telemetry data through the following pipeline:

```text
Telemetry Generator
        ↓
Anomaly Generator
        ↓
Apache Kafka
        ↓
Real-Time ML Consumer
        ↓
Autoencoder
        ↓
Anomaly Score
        ↓
Threshold Detection
        ↓
Root-Cause Feature Analysis
        ↓
Alert Logger
        ↓
FastAPI
        ↓
Live Dashboard