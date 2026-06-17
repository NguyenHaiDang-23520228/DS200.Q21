import json
import os
import socket
from datetime import datetime

from common import create_server, recv_json, send_json
from config import Config


def ensure_results_dir() -> None:
    os.makedirs(Config.results_dir, exist_ok=True)


def save_result(result: dict) -> str:
    ensure_results_dir()
    frame_id = result.get("frame_id", "unknown")
    filename = os.path.join(Config.results_dir, f"frame_{frame_id:04d}.json")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    log_path = os.path.join(Config.results_dir, "detections.ndjson")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return filename


def update_summary() -> None:
    ensure_results_dir()
    log_path = os.path.join(Config.results_dir, "detections.ndjson")
    summary_path = os.path.join(Config.results_dir, "summary.json")

    if not os.path.exists(log_path):
        return

    records = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    person_counts = [item.get("person_count", 0) for item in records]
    summary = {
        "total_frames": len(records),
        "total_persons_detected": sum(person_counts),
        "average_person_count": sum(person_counts) / len(person_counts),
        "max_person_count": max(person_counts),
        "updated_at": datetime.now().isoformat(),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def handle_client(connection: socket.socket, address) -> None:
    print(f"[Storage] Connected from {address}")
    try:
        while True:
            payload = recv_json(connection)
            if payload is None:
                break

            if payload.get("type") == "shutdown":
                send_json(connection, {"status": "ok", "message": "storage shutting down"})
                break

            payload["stored_at"] = datetime.now().isoformat()
            saved_path = save_result(payload)
            print(
                f"[Storage] Saved frame {payload.get('frame_id')} | "
                f"person_count={payload.get('person_count')} -> {saved_path}"
            )
            send_json(connection, {"status": "ok", "frame_id": payload.get("frame_id")})
    finally:
        connection.close()
        update_summary()
        print("[Storage] Connection closed. Summary updated.")


def main() -> None:
    ensure_results_dir()
    server = create_server(Config.storage_host, Config.storage_port)
    print(f"[Storage] Waiting for results on {Config.storage_host}:{Config.storage_port}")

    try:
        while True:
            connection, address = server.accept()
            handle_client(connection, address)
    except KeyboardInterrupt:
        print("\n[Storage] Stopped.")
    finally:
        server.close()
        update_summary()


if __name__ == "__main__":
    main()
