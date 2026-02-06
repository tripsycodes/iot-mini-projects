import os
import csv
import time
from collections import deque
from flask import Flask, jsonify

FIFO_PATH = "soil.sock"
CSV_PATH = "soil_data.csv"

FIELDNAMES = [
    "time",
    "moisture",
    "temperature",
    "ec",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "salinity",
    "tds",
]

# Keep last 5 readings in memory
latest_readings = deque(maxlen=5)

app = Flask(__name__)

@app.route("/latest", methods=["GET"])
def get_latest():
    return jsonify(list(latest_readings))


def fifo_reader():
    if not os.path.exists(FIFO_PATH):
        raise FileNotFoundError(f"FIFO not found: {FIFO_PATH}")

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(FIELDNAMES)

        print("Waiting for FIFO writer...")

        while True:
            with open(FIFO_PATH, "r") as fifo:
                print("FIFO opened")

                for line in fifo:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(",")
                    if len(parts) != len(FIELDNAMES):
                        print("Malformed line:", line)
                        continue

                    try:
                        row = {
                            "time": int(parts[0]),
                            "moisture": float(parts[1]),
                            "temperature": float(parts[2]),
                            "ec": int(parts[3]),
                            "ph": float(parts[4]),
                            "nitrogen": int(parts[5]),
                            "phosphorus": int(parts[6]),
                            "potassium": int(parts[7]),
                            "salinity": float(parts[8]),
                            "tds": int(parts[9]),
                        }
                    except ValueError:
                        print("Parse error:", line)
                        continue

                    # Persist to CSV
                    writer.writerow(row.values())
                    csv_file.flush()

                    # Update in-memory cache
                    latest_readings.append(row)

                    print("Logged:", row)

            # Writer disconnected
            time.sleep(0.1)


if __name__ == "__main__":
    from threading import Thread

    # Start FIFO reader thread
    t = Thread(target=fifo_reader, daemon=True)
    t.start()

    # Start API server
    print("Starting API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)

