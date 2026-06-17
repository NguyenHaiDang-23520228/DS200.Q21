import base64
import socket
import time
from datetime import datetime

import cv2
import numpy as np

from common import close_quietly, connect_tcp, create_server, recv_json, send_json
from config import Config


class PersonDetector:
    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image: np.ndarray) -> tuple:
        if image is None or image.size == 0:
            return 0, []

        resized = image
        if max(image.shape[:2]) > 800:
            scale = 800 / max(image.shape[:2])
            resized = cv2.resize(image, None, fx=scale, fy=scale)

        rects, _ = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        boxes = []
        for x, y, w, h in rects:
            boxes.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            })

        return len(boxes), boxes


def decode_frame(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def forward_to_storage(result: dict) -> None:
    storage_connection = None
    try:
        storage_connection = connect_tcp(Config.storage_host, Config.storage_port)
        send_json(storage_connection, result)
        response = recv_json(storage_connection)
        if response and response.get("status") == "ok":
            print(f"[Processing] Stored frame {result.get('frame_id')} successfully")
        else:
            print(f"[Processing] Storage response: {response}")
    except ConnectionRefusedError:
        print("[Processing] ERROR: Storage server is not running.")
        raise
    finally:
        close_quietly(storage_connection)


def process_frame(payload: dict, detector: PersonDetector) -> dict:
    started = time.time()
    image = decode_frame(payload["image_base64"])
    person_count, bounding_boxes = detector.detect(image)
    elapsed_ms = int((time.time() - started) * 1000)

    return {
        "type": "detection_result",
        "frame_id": payload.get("frame_id"),
        "source_timestamp": payload.get("timestamp"),
        "processed_at": datetime.now().isoformat(),
        "person_count": person_count,
        "bounding_boxes": bounding_boxes,
        "image_width": payload.get("width"),
        "image_height": payload.get("height"),
        "processing_time_ms": elapsed_ms,
    }


def handle_camera_client(connection: socket.socket, address, detector: PersonDetector) -> None:
    print(f"[Processing] Camera server connected from {address}")
    try:
        while True:
            payload = recv_json(connection)
            if payload is None:
                break

            if payload.get("type") == "shutdown":
                send_json(connection, {"status": "ok", "message": "processing complete"})
                break

            if payload.get("type") != "frame":
                continue

            result = process_frame(payload, detector)
            print(
                f"[Processing] Frame {result['frame_id']} | "
                f"person_count={result['person_count']} | "
                f"{result['processing_time_ms']} ms"
            )
            forward_to_storage(result)
            send_json(connection, {
                "status": "ok",
                "frame_id": result["frame_id"],
                "person_count": result["person_count"],
                "bounding_boxes": result["bounding_boxes"],
            })
    finally:
        connection.close()
        print("[Processing] Camera connection closed.")


def main() -> None:
    detector = PersonDetector()
    server = create_server(Config.processing_host, Config.processing_port)
    print(
        f"[Processing] Waiting for camera frames on "
        f"{Config.processing_host}:{Config.processing_port}"
    )

    try:
        while True:
            connection, address = server.accept()
            handle_camera_client(connection, address, detector)
    except KeyboardInterrupt:
        print("\n[Processing] Stopped.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
