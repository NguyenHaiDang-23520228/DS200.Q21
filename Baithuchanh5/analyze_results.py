import json
import os
from collections import Counter

from config import Config


def main() -> None:
    log_path = os.path.join(Config.results_dir, "detections.ndjson")
    if not os.path.exists(log_path):
        print("No detection log found. Run the system first.")
        return

    records = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    counts = Counter(item.get("person_count", 0) for item in records)
    total_boxes = sum(len(item.get("bounding_boxes", [])) for item in records)

    print("=== Batch Analytics on Stored Detection Results ===")
    print(f"Total frames analyzed: {len(records)}")
    print(f"Total bounding boxes: {total_boxes}")
    print("Person count distribution:")
    for person_count, frequency in sorted(counts.items()):
        print(f"  {person_count} person(s): {frequency} frame(s)")


if __name__ == "__main__":
    main()
