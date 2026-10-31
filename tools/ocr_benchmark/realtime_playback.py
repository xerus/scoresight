"""Measure source-time drift and OCR completion age during actual file playback."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from camera_info import CameraInfo
from camera_thread import TimerThread
from defaults import (
    basketball_settings_for_box,
    default_info_for_box_name,
    normalize_settings_dict,
)
from storage import TextDetectionTargetMemoryStorage
from text_detection_target import TextDetectionTarget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--start-frame", type=int, default=1800)
    parser.add_argument("--camera-module", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QCoreApplication([])
    thread_class = TimerThread
    if args.camera_module:
        spec = importlib.util.spec_from_file_location(
            "baseline_camera", args.camera_module
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        thread_class = module.TimerThread
    storage = TextDetectionTargetMemoryStorage()
    for name, roi in (
        ("Home Score", (140, 445, 120, 90)),
        ("Away Score", (430, 430, 105, 90)),
        ("Time", (255, 350, 155, 79)),
    ):
        settings = normalize_settings_dict(
            dict(
                basketball_settings_for_box(name),
                box_ocr_model=6,
                check_led=True,
                clock_input=1 if name == "Time" else 0,
            ),
            default_info_for_box_name(name),
        )
        storage.add_item(TextDetectionTarget(*roi, name, settings))
    worker = thread_class(
        CameraInfo(
            str(args.video),
            str(args.video),
            str(args.video),
            CameraInfo.CameraType.FILE,
        ),
        storage,
    )
    worker.textDetector.setOcrModel(6)
    worker.update_frame_interval = 200  # request five OCR updates/s
    original_connect = worker.connect_video_capture

    def connect():
        ok = original_connect()
        if ok:
            worker.video_capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        return ok

    worker.connect_video_capture = connect
    observations = []
    original_detect = worker.textDetector.detect_multi_text

    def detect(*values, **kwargs):
        start = time.monotonic()
        position = worker.observation_time_for_file()
        result = original_detect(*values, **kwargs)
        observations.append(
            dict(source_time=position, started_at=start, completed_at=time.monotonic())
        )
        return result

    worker.textDetector.detect_multi_text = detect
    previews = []
    worker.update_signal.connect(lambda frame: previews.append(time.monotonic()))
    loop = QEventLoop()
    worker.finished.connect(loop.quit)

    def stop():
        worker.should_stop = True

    QTimer.singleShot(round(args.seconds * 1000), stop)
    worker.start()
    loop.exec()
    worker.wait()
    if len(observations) < 2:
        raise RuntimeError("Insufficient OCR observations")
    first, last = observations[0], observations[-1]
    age = [
        row["completed_at"]
        - first["started_at"]
        - (row["source_time"] - first["source_time"])
        for row in observations
    ]
    duration = last["completed_at"] - first["started_at"]
    output = dict(
        video=args.video.name,
        start_frame=args.start_frame,
        requested_ocr_hz=5,
        fields=3,
        wall_seconds=duration,
        source_seconds=last["source_time"] - first["source_time"],
        output_age_seconds=dict(
            median=statistics.median(age), maximum=max(age), last=age[-1]
        ),
        processing_ms_median=statistics.median(
            (row["completed_at"] - row["started_at"]) * 1000 for row in observations
        ),
        ocr_updates=len(observations),
        preview_updates=len(previews),
        observations=observations,
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "observations"}))
    del worker


if __name__ == "__main__":
    main()
