import base64
import glob
import os
import socket
import time
from datetime import datetime

import cv2
import numpy as np

from common import close_quietly, connect_tcp, recv_json, send_json
from config import Config


def encode_frame(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Cannot encode frame to JPEG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def build_frame_payload(frame_id: int, image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    return {
        "type": "frame",
        "frame_id": frame_id,
        "timestamp": datetime.now().isoformat(),
        "width": width,
        "height": height,
        "image_base64": encode_frame(image),
    }


def iter_webcam_frames(max_frames: int):
    capture = cv2.VideoCapture(Config.camera_index)
    if not capture.isOpened():
        return

    frame_id = 1
    while frame_id <= max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame_id, frame
        frame_id += 1
        time.sleep(Config.frame_interval_sec)
    capture.release()


def iter_video_frames(max_frames: int):
    if not os.path.exists(Config.input_video):
        return

    capture = cv2.VideoCapture(Config.input_video)
    if not capture.isOpened():
        return

    frame_id = 1
    while frame_id <= max_frames:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
            if not ok:
                break
        yield frame_id, frame
        frame_id += 1
        time.sleep(Config.frame_interval_sec)
    capture.release()


def iter_image_frames(max_frames: int):
    patterns = [
        os.path.join(Config.input_image_dir, "*.jpg"),
        os.path.join(Config.input_image_dir, "*.jpeg"),
        os.path.join(Config.input_image_dir, "*.png"),
    ]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(sorted(glob.glob(pattern)))

    if not image_paths:
        return

    frame_id = 1
    index = 0
    while frame_id <= max_frames:
        image = cv2.imread(image_paths[index % len(image_paths)])
        if image is None:
            break
        yield frame_id, image
        frame_id += 1
        index += 1
        time.sleep(Config.frame_interval_sec)


def iter_synthetic_frames(max_frames: int):
    frame_id = 1
    while frame_id <= max_frames:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            image,
            f"Synthetic Frame {frame_id}",
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(image, (200, 120), (320, 360), (0, 255, 0), 2)
        yield frame_id, image
        frame_id += 1
        time.sleep(Config.frame_interval_sec)


def frame_source(max_frames: int):
    for source in (iter_video_frames, iter_image_frames, iter_webcam_frames, iter_synthetic_frames):
        frames = list(source(max_frames))
        if frames:
            print(f"[Camera] Using source: {source.__name__} ({len(frames)} frames)")
            yield from frames
            return

    print("[Camera] No frame source available.")
    yield from iter_synthetic_frames(max_frames)


def main() -> None:
    connection = None
    try:
        print(
            f"[Camera] Connecting to processing server "
            f"{Config.processing_host}:{Config.processing_port}..."
        )
        connection = connect_tcp(Config.processing_host, Config.processing_port)
        print("[Camera] Connected. Streaming frames...")

        for frame_id, image in frame_source(Config.max_frames):
            payload = build_frame_payload(frame_id, image)
            send_json(connection, payload)
            print(f"[Camera] Sent frame {frame_id}")

            response = recv_json(connection)
            if response:
                print(
                    f"[Camera] Frame {response.get('frame_id')} -> "
                    f"person_count={response.get('person_count')}"
                )

        send_json(connection, {"type": "shutdown"})
        print("[Camera] Finished streaming.")
    except ConnectionRefusedError:
        print("[Camera] ERROR: Processing server is not running.")
    except BrokenPipeError:
        print("[Camera] Connection closed by processing server.")
    except Exception as error:
        print(f"[Camera] Exception: {error}")
    finally:
        close_quietly(connection)


if __name__ == "__main__":
    main()
