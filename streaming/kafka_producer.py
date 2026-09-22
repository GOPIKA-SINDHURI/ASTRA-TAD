import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


KAFKA_SERVER = "localhost:9092"
TOPIC = "telemetry"


producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)


def generate_telemetry():
    """
    Generate one spacecraft-like telemetry record.
    """

    telemetry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "temperature": round(random.uniform(24.0, 26.0), 3),
        "voltage": round(random.uniform(27.5, 28.5), 3),
        "current": round(random.uniform(2.8, 3.2), 3),
        "pressure": round(random.uniform(99.0, 103.0), 3),
        "vibration": round(random.uniform(0.07, 0.13), 3),
        "rpm": round(random.uniform(2950, 3050), 2),
        "battery": round(random.uniform(86.0, 90.0), 3),

        "anomaly_type": "normal"
    }

    # Approximately 5% chance of injecting an anomaly
    if random.random() < 0.05:

        anomaly = random.choice([
            "temperature_spike",
            "voltage_drop",
            "high_vibration",
            "battery_degradation"
        ])

        telemetry["anomaly_type"] = anomaly

        if anomaly == "temperature_spike":
            telemetry["temperature"] = round(
                random.uniform(40.0, 55.0), 3
            )

        elif anomaly == "voltage_drop":
            telemetry["voltage"] = round(
                random.uniform(15.0, 22.0), 3
            )

        elif anomaly == "high_vibration":
            telemetry["vibration"] = round(
                random.uniform(0.5, 1.0), 3
            )

        elif anomaly == "battery_degradation":
            telemetry["battery"] = round(
                random.uniform(65.0, 75.0), 3
            )

    return telemetry


def main():

    print("=" * 60)
    print("ASTRA-TAD REAL-TIME TELEMETRY PRODUCER")
    print("=" * 60)

    print(f"Kafka Server : {KAFKA_SERVER}")
    print(f"Kafka Topic  : {TOPIC}")
    print("Sending telemetry every 1 second...")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:

        while True:

            telemetry = generate_telemetry()

            producer.send(TOPIC, value=telemetry)
            producer.flush()

            print(
                f"[{telemetry['timestamp']}] "
                f"T={telemetry['temperature']} | "
                f"V={telemetry['voltage']} | "
                f"I={telemetry['current']} | "
                f"P={telemetry['pressure']} | "
                f"Vib={telemetry['vibration']} | "
                f"RPM={telemetry['rpm']} | "
                f"Battery={telemetry['battery']} | "
                f"Anomaly={telemetry['anomaly_type']}"
            )

            time.sleep(1)

    except KeyboardInterrupt:

        print("\nStopping telemetry producer...")

    finally:

        producer.close()
        print("Producer stopped.")


if __name__ == "__main__":
    main()