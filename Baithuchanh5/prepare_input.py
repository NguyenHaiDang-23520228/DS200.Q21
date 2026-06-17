import os

import cv2
import numpy as np

from config import Config


def ensure_input_dir() -> None:
    os.makedirs(Config.input_image_dir, exist_ok=True)
    os.makedirs("input", exist_ok=True)


def generate_sample_video() -> None:
    ensure_input_dir()
    output_path = Config.input_video
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (640, 480),
    )

    for index in range(15):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(
            frame,
            f"Fecom Camera Frame {index + 1}",
            (60, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (120, 120), (220, 420), (0, 255, 255), 2)
        cv2.rectangle(frame, (360, 140), (470, 430), (255, 128, 0), 2)
        writer.write(frame)

        image_path = os.path.join(Config.input_image_dir, f"frame_{index + 1:02d}.jpg")
        cv2.imwrite(image_path, frame)

    writer.release()
    print(f"Generated {output_path} and sample images in {Config.input_image_dir}/")


if __name__ == "__main__":
    generate_sample_video()
