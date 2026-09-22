import json
import time
import random
from datetime import datetime

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

STREAM_INTERVAL = 1.0

ANOMALY_PROBABILITY = 0.05


# ============================================================
# NORMAL TELEMETRY RANGES
# ============================================================

NORMAL_RANGES = {

    "temperature": (24.0, 26.0),

    "voltage": (27.5, 28.5),

    "current": (2.8, 3.2),

    "pressure": (99.0, 103.0),

    "vibration": (0.07, 0.13),

    "rpm": (2950, 3050),

    "battery": (86.0, 90.0)
}


# ============================================================
# GENERATE NORMAL TELEMETRY
# ============================================================

def generate_normal():

    telemetry = {

        "timestamp":
            datetime.now().isoformat(),

        "temperature":
            random.uniform(
                *NORMAL_RANGES["temperature"]
            ),

        "voltage":
            random.uniform(
                *NORMAL_RANGES["voltage"]
            ),

        "current":
            random.uniform(
                *NORMAL_RANGES["current"]
            ),

        "pressure":
            random.uniform(
                *NORMAL_RANGES["pressure"]
            ),

        "vibration":
            random.uniform(
                *NORMAL_RANGES["vibration"]
            ),

        "rpm":
            random.uniform(
                *NORMAL_RANGES["rpm"]
            ),

        "battery":
            random.uniform(
                *NORMAL_RANGES["battery"]
            )
    }

    return telemetry


# ============================================================
# GENERATE ANOMALY
# ============================================================

def inject_anomaly(telemetry):

    anomaly_type = random.choice([

        "temperature_spike",

        "voltage_drop",

        "high_vibration",

        "battery_degradation"
    ])


    if anomaly_type == "temperature_spike":

        telemetry["temperature"] = random.uniform(
            40,
            55
        )


    elif anomaly_type == "voltage_drop":

        telemetry["voltage"] = random.uniform(
            15,
            22
        )


    elif anomaly_type == "high_vibration":

        telemetry["vibration"] = random.uniform(
            0.5,
            1.0
        )


    elif anomaly_type == "battery_degradation":

        telemetry["battery"] = random.uniform(
            65,
            75
        )


    telemetry["anomaly_type"] = anomaly_type

    return telemetry


# ============================================================
# GENERATE TELEMETRY EVENT
# ============================================================

def generate_event():

    telemetry = generate_normal()

    is_anomaly = (
        random.random()
        <
        ANOMALY_PROBABILITY
    )


    if is_anomaly:

        telemetry = inject_anomaly(
            telemetry
        )

        telemetry["anomaly"] = 1

    else:

        telemetry["anomaly"] = 0

        telemetry["anomaly_type"] = "normal"


    return telemetry


# ============================================================
# MAIN STREAM
# ============================================================

def main():

    print("=" * 60)

    print(
        "ASTRA-TAD REAL-TIME TELEMETRY PRODUCER"
    )

    print("=" * 60)

    print(
        "\nGenerating telemetry..."
    )

    print(
        "Press CTRL+C to stop.\n"
    )


    event_number = 0


    try:

        while True:

            event_number += 1

            telemetry = generate_event()


            print(
                json.dumps(
                    telemetry,
                    indent=2
                )
            )


            print(
                f"Event #{event_number}"
            )


            if telemetry["anomaly"] == 1:

                print(
                    "🚨 ANOMALY INJECTED:",
                    telemetry[
                        "anomaly_type"
                    ]
                )

            else:

                print(
                    "✓ Normal telemetry"
                )


            print("-" * 60)


            time.sleep(
                STREAM_INTERVAL
            )


    except KeyboardInterrupt:

        print(
            "\n\nTelemetry stream stopped."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()