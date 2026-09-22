import json
import time

import numpy as np
import pandas as pd
from kafka import KafkaProducer


# ============================================================
# ASTRA-TAD TELEMETRY GENERATOR
# ============================================================

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "telemetry"

NUM_RECORDS = 10000
SEED = 42


def generate_telemetry(n_samples=10000, seed=42):
    """
    Generate simulated healthy multivariate telemetry.
    """

    np.random.seed(seed)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=n_samples,
        freq="s"
    )

    # --------------------------------------------------------
    # Base telemetry signals
    # --------------------------------------------------------

    temperature = (
        25
        + 0.8 * np.sin(np.linspace(0, 20 * np.pi, n_samples))
        + np.random.normal(0, 0.5, n_samples)
    )

    voltage = (
        28
        + 0.3 * np.sin(np.linspace(0, 10 * np.pi, n_samples))
        + np.random.normal(0, 0.2, n_samples)
    )

    current = (
        3
        + 0.15 * np.sin(np.linspace(0, 15 * np.pi, n_samples))
        + np.random.normal(0, 0.1, n_samples)
    )

    pressure = (
        101
        + 0.5 * np.sin(np.linspace(0, 8 * np.pi, n_samples))
        + np.random.normal(0, 0.3, n_samples)
    )

    vibration = (
        0.1
        + 0.02 * np.sin(np.linspace(0, 30 * np.pi, n_samples))
        + np.random.normal(0, 0.01, n_samples)
    )

    rpm = (
        3000
        + 50 * np.sin(np.linspace(0, 12 * np.pi, n_samples))
        + np.random.normal(0, 30, n_samples)
    )

    battery = (
        90
        - np.linspace(0, 5, n_samples)
        + np.random.normal(0, 0.1, n_samples)
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    data = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "voltage": voltage,
        "current": current,
        "pressure": pressure,
        "vibration": vibration,
        "rpm": rpm,
        "battery": battery
    })

    return data


def create_kafka_producer():

    print("\nConnecting to Kafka...")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value:
            json.dumps(value).encode("utf-8")
    )

    print(f"✓ Connected to Kafka: {KAFKA_SERVER}")

    return producer


def publish_telemetry(df, producer):

    print(f"✓ Publishing telemetry to topic: {KAFKA_TOPIC}")
    print()

    for index, row in df.iterrows():

        telemetry = {
            "timestamp": str(row["timestamp"]),
            "temperature": float(row["temperature"]),
            "voltage": float(row["voltage"]),
            "current": float(row["current"]),
            "pressure": float(row["pressure"]),
            "vibration": float(row["vibration"]),
            "rpm": float(row["rpm"]),
            "battery": float(row["battery"])
        }

        producer.send(
            KAFKA_TOPIC,
            value=telemetry
        )

        # Print progress
        if (index + 1) % 100 == 0:
            producer.flush()

            print(
                f"Published {index + 1}/{len(df)} records"
            )

        # Small delay to simulate real-time telemetry
        time.sleep(0.05)

    producer.flush()

    print()
    print("✓ All telemetry published successfully!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("       ASTRA-TAD REAL-TIME TELEMETRY GENERATOR")
    print("=" * 55)

    print("\nGenerating healthy telemetry...")

    df = generate_telemetry(
        n_samples=NUM_RECORDS,
        seed=SEED
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_path = "data/synthetic/healthy_telemetry.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"✓ Generated records : {len(df)}")
    print(f"✓ CSV saved to      : {output_path}")

    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    print("\nFirst 5 records:")
    print(df.head())

    # --------------------------------------------------------
    # Kafka
    # --------------------------------------------------------

    try:

        producer = create_kafka_producer()

        publish_telemetry(
            df,
            producer
        )

        producer.close()

    except Exception as e:

        print()
        print("❌ Kafka publishing failed!")
        print(f"Error: {e}")

    print()
    print("=" * 55)
    print("       TELEMETRY GENERATION COMPLETED")
    print("=" * 55)