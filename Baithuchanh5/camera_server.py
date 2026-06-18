import argparse
import base64
import glob
import os
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


def draw_boxes(image: np.ndarray, response: dict) -> np.ndarray:
    output = image.copy()
    person_count = response.get("person_count", 0)

    for box in response.get("bounding_boxes", []):
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["width"])
        h = int(box["height"])
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"People: {person_count} | Frame: {response.get('frame_id')}"
    cv2.rectangle(output, (10, 10), (320, 45), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def save_annotated(frame_id: int, image: np.ndarray) -> None:
    os.makedirs(Config.annotated_dir, exist_ok=True)
    path = os.path.join(Config.annotated_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(path, image)


def iter_webcam_frames(max_frames: int, realtime: bool):
    capture = cv2.VideoCapture(Config.camera_index)
    if not capture.isOpened():
        return

    frame_id = 1
    while realtime or frame_id <= max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame_id, frame
        frame_id += 1
        if not realtime:
            time.sleep(Config.frame_interval_sec)
    capture.release()


def iter_video_frames(max_frames: int, realtime: bool, video_path: str):
    if not os.path.exists(video_path):
        return

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return

    frame_id = 1
    while realtime or frame_id <= max_frames:
        ok, frame = capture.read()
        if not ok:
            if realtime:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    break
            else:
                break
        yield frame_id, frame
        frame_id += 1
        if not realtime:
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


def frame_source(source: str, max_frames: int, realtime: bool, video_path: str):
    if source == "webcam":
        frames = list(iter_webcam_frames(max_frames, realtime))
        if frames:
            print(f"[Camera] Using webcam ({len(frames)} frames)")
            yield from frames
            return

    if source in ("video", "auto"):
        frames = list(iter_video_frames(max_frames, realtime, video_path))
        if frames:
            print(f"[Camera] Using video: {video_path} ({len(frames)} frames)")
            yield from frames
            return

    if source in ("images", "auto"):
        frames = list(iter_image_frames(max_frames))
        if frames:
            print(f"[Camera] Using images ({len(frames)} frames)")
            yield from frames
            return

    if source == "webcam" or source == "auto":
        frames = list(iter_webcam_frames(max_frames, realtime))
        if frames:
            print(f"[Camera] Using webcam ({len(frames)} frames)")
            yield from frames
            return

    print("[Camera] Fallback to synthetic frames.")
    yield from iter_synthetic_frames(max_frames)


def parse_args():
    parser = argparse.ArgumentParser(description="Camera server - stream frames to processing server")
    parser.add_argument(
        "--source",
        choices=["auto", "webcam", "video", "images"],
        default="auto",
        help="Frame source: auto | webcam | video | images",
    )
    parser.add_argument(
        "--video",
        default=Config.input_video,
        help="Path to video file when --source video",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Stream continuously until you press Q in the preview window",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show preview window with bounding boxes returned by processing server",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=Config.max_frames,
        help="Number of frames to send (ignored in --realtime mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = None

    try:
        print(
            f"[Camera] Connecting to processing server "
            f"{Config.processing_host}:{Config.processing_port}..."
        )
        connection = connect_tcp(Config.processing_host, Config.processing_port)
        print("[Camera] Connected. Streaming frames...")
        print("[Camera] Press Q in preview window to stop (when --show is enabled).")

        for frame_id, image in frame_source(
            args.source,
            args.max_frames,
            args.realtime,
            args.video,
        ):
            payload = build_frame_payload(frame_id, image)
            send_json(connection, payload)
            print(f"[Camera] Sent frame {frame_id}")

            response = recv_json(connection)
            if not response:
                break

            print(
                f"[Camera] Frame {response.get('frame_id')} -> "
                f"person_count={response.get('person_count')} | "
                f"boxes={len(response.get('bounding_boxes', []))}"
            )

            if args.show or args.realtime:
                annotated = draw_boxes(image, response)
                save_annotated(frame_id, annotated)
                cv2.imshow("Baithuchanh5 - Real-time People Counting", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Camera] Stopped by user (Q pressed).")
                    break

        send_json(connection, {"type": "shutdown"})
        print("[Camera] Finished streaming.")
    except ConnectionRefusedError:
        print("[Camera] ERROR: Processing server is not running.")
        print("Start storage_server.py and processing_server.py first.")
    except BrokenPipeError:
        print("[Camera] Connection closed by processing server.")
    except Exception as error:
        print(f"[Camera] Exception: {error}")
    finally:
        close_quietly(connection)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
