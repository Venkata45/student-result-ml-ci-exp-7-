import json
import sys

def check_gate():
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("FAIL: metrics.json not found.")
        sys.exit(1)

    acc = metrics.get("accuracy", 0.0)
    threshold = 0.50

    if acc >= threshold:
        print(f"PASS: Accuracy {acc:.4f} meets threshold of {threshold}")
        sys.exit(0)
    else:
        print(f"FAIL: Accuracy {acc:.4f} below threshold of {threshold}")
        sys.exit(1)

if __name__ == "__main__":
    check_gate()
