import glob
import json
import os

import cv2

from config import Config


def load_result(frame_id: int) -> dict:
    path = os.path.join(Config.results_dir, f"frame_{frame_id:04d}.json")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_source_image(frame_id: int):
    candidates = [
        os.path.join(Config.input_image_dir, f"frame_{frame_id:02d}.jpg"),
        Config.input_video,
    ]
    for path in candidates:
        if os.path.exists(path):
            if path.endswith(".mp4"):
                capture = cv2.VideoCapture(path)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                ok, frame = capture.read()
                capture.release()
                if ok:
                    return frame
            else:
                image = cv2.imread(path)
                if image is not None:
                    return image
    return None


def draw_and_save(frame_id: int, result: dict, image) -> str:
    os.makedirs(Config.annotated_dir, exist_ok=True)
    output = image.copy()

    for box in result.get("bounding_boxes", []):
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["width"])
        h = int(box["height"])
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"People: {result.get('person_count', 0)}"
    cv2.putText(
        output,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    out_path = os.path.join(Config.annotated_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(out_path, output)
    return out_path


def main() -> None:
    result_files = sorted(glob.glob(os.path.join(Config.results_dir, "frame_*.json")))
    if not result_files:
        print("No results found. Run the system first.")
        return

    print("=== Visualize Detection Results ===")
    for result_path in result_files:
        frame_id = int(os.path.basename(result_path).split("_")[1].split(".")[0])
        result = load_result(frame_id)
        image = find_source_image(frame_id)
        if image is None:
            print(f"Skip frame {frame_id}: source image not found")
            continue
        saved = draw_and_save(frame_id, result, image)
        print(f"Saved {saved} | person_count={result.get('person_count')}")

    print(f"\nAnnotated images are in {Config.annotated_dir}/")


if __name__ == "__main__":
    main()
