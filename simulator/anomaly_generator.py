import json
import time

import numpy as np
import pandas as pd
from kafka import KafkaProducer


# ============================================================
# ASTRA-TAD ANOMALY GENERATOR
# ============================================================

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "telemetry"

INPUT_PATH = "data/synthetic/healthy_telemetry.csv"
OUTPUT_PATH = "data/synthetic/telemetry_with_anomalies.csv"

SEED = 42


# ============================================================
# ANOMALY INJECTION
# ============================================================

def inject_anomalies(df, seed=42):
    """
    Inject multiple types of synthetic anomalies into
    healthy multivariate telemetry.

    Returns:
        DataFrame containing anomaly labels and anomaly types.
    """

    np.random.seed(seed)

    data = df.copy()

    # Ground-truth labels
    data["anomaly"] = 0
    data["anomaly_type"] = "normal"

    n = len(data)

    # --------------------------------------------------------
    # 1. TEMPERATURE SPIKE
    # --------------------------------------------------------

    start = int(n * 0.10)
    end = start + 80

    data.loc[start:end, "temperature"] += np.linspace(
        5,
        25,
        end - start + 1
    )

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "temperature_spike"

    # --------------------------------------------------------
    # 2. VOLTAGE DROP
    # --------------------------------------------------------

    start = int(n * 0.25)
    end = start + 100

    data.loc[start:end, "voltage"] -= np.linspace(
        2,
        7,
        end - start + 1
    )

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "voltage_drop"

    # --------------------------------------------------------
    # 3. SENSOR DRIFT
    # --------------------------------------------------------

    start = int(n * 0.40)
    end = start + 250

    drift = np.linspace(
        0,
        8,
        end - start + 1
    )

    data.loc[start:end, "pressure"] += drift

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "sensor_drift"

    # --------------------------------------------------------
    # 4. STUCK SENSOR
    # --------------------------------------------------------

    start = int(n * 0.55)
    end = start + 120

    stuck_value = data.loc[start, "current"]

    data.loc[start:end, "current"] = stuck_value

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "stuck_sensor"

    # --------------------------------------------------------
    # 5. HIGH VIBRATION
    # --------------------------------------------------------

    start = int(n * 0.68)
    end = start + 100

    data.loc[start:end, "vibration"] += np.random.normal(
        0,
        0.15,
        end - start + 1
    )

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "high_vibration"

    # --------------------------------------------------------
    # 6. CORRELATED MULTI-SENSOR ANOMALY
    # --------------------------------------------------------

    start = int(n * 0.78)
    end = start + 100

    data.loc[start:end, "temperature"] += 10
    data.loc[start:end, "current"] += 1
    data.loc[start:end, "vibration"] += 0.08

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "correlated_failure"

    # --------------------------------------------------------
    # 7. BATTERY DEGRADATION
    # --------------------------------------------------------

    start = int(n * 0.88)
    end = min(start + 300, n - 1)

    degradation = np.linspace(
        0,
        20,
        end - start + 1
    )

    data.loc[start:end, "battery"] -= degradation

    data.loc[start:end, "anomaly"] = 1
    data.loc[start:end, "anomaly_type"] = "battery_degradation"

    return data


# ============================================================
# KAFKA PRODUCER
# ============================================================

def create_kafka_producer():

    print()
    print("Connecting to Kafka...")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value:
            json.dumps(value).encode("utf-8")
    )

    print(f"✓ Connected to Kafka: {KAFKA_SERVER}")

    return producer


# ============================================================
# PUBLISH TELEMETRY
# ============================================================

def publish_telemetry(df, producer):

    print()
    print(f"✓ Publishing records to Kafka topic: {KAFKA_TOPIC}")
    print()

    total = len(df)

    for index, row in df.iterrows():

        telemetry = {
            "timestamp": str(row["timestamp"]),

            "temperature": float(row["temperature"]),
            "voltage": float(row["voltage"]),
            "current": float(row["current"]),
            "pressure": float(row["pressure"]),
            "vibration": float(row["vibration"]),
            "rpm": float(row["rpm"]),
            "battery": float(row["battery"]),

            # Ground truth
            "anomaly": int(row["anomaly"]),
            "anomaly_type": str(row["anomaly_type"])
        }

        producer.send(
            KAFKA_TOPIC,
            value=telemetry
        )

        # Flush and show progress every 100 records
        if (index + 1) % 100 == 0:

            producer.flush()

            print(
                f"Published {index + 1}/{total} records"
            )

        # Simulate real-time telemetry
        time.sleep(0.05)

    producer.flush()

    print()
    print("✓ All anomaly telemetry published successfully!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("       ASTRA-TAD ANOMALY GENERATOR")
    print("=" * 55)

    # --------------------------------------------------------
    # Load healthy telemetry
    # --------------------------------------------------------

    print()
    print("Loading healthy telemetry...")

    df = pd.read_csv(INPUT_PATH)

    print(f"✓ Loaded {len(df)} healthy records")

    # --------------------------------------------------------
    # Inject anomalies
    # --------------------------------------------------------

    print()
    print("Injecting anomalies...")

    anomaly_df = inject_anomalies(
        df,
        seed=SEED
    )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    anomaly_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display statistics
    # --------------------------------------------------------

    print()
    print("Anomaly distribution:")

    print(
        anomaly_df["anomaly_type"]
        .value_counts()
    )

    print()
    print(
        f"Total anomalous records : "
        f"{anomaly_df['anomaly'].sum()}"
    )

    print(
        f"Total normal records    : "
        f"{(anomaly_df['anomaly'] == 0).sum()}"
    )

    print()
    print(f"✓ Dataset saved to: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # Kafka publishing
    # --------------------------------------------------------

    try:

        producer = create_kafka_producer()

        publish_telemetry(
            anomaly_df,
            producer
        )

        producer.close()

    except Exception as e:

        print()
        print("❌ Kafka publishing failed!")
        print(f"Error: {e}")

    print()
    print("=" * 55)
    print("       ANOMALY TEST COMPLETED")
    print("=" * 55)