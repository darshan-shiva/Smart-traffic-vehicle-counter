"""Unit and Integration Tests for VehicleTracker Module.

Author: Darshan S.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.tracker import (
    BaseTracker,
    ByteTrackTracker,
    BotSortTracker,
    VehicleTracker,
)


def test_tracker_initialization() -> None:
    """Verify tracker initialization and factory dispatching."""
    detector = VehicleDetector(model_path="models/yolo11n.pt")

    bytetrack = VehicleTracker(tracker_type="bytetrack", detector=detector)
    assert isinstance(bytetrack.tracker, ByteTrackTracker)
    assert isinstance(bytetrack.tracker, BaseTracker)

    botsort = VehicleTracker(tracker_type="botsort", detector=detector)
    assert isinstance(botsort.tracker, BotSortTracker)
    assert isinstance(botsort.tracker, BaseTracker)

    try:
        VehicleTracker(tracker_type="unsupported_tracker")
        assert False, "Should have raised ValueError for invalid tracker type"
    except ValueError:
        pass

    print("✓ Tracker abstraction and factory initialization verified.")


def test_track_data_structure() -> None:
    """Verify that returned tracks have all required keys and correct data types."""
    video_path = PROJECT_ROOT / "assets" / "traffic_sample.mp4"
    assert video_path.exists(), f"Missing video: {video_path}"

    detector = VehicleDetector(model_path="models/yolo11n.pt", confidence_threshold=0.35)
    tracker = VehicleTracker(tracker_type="bytetrack", detector=detector)

    cap = cv2.VideoCapture(str(video_path))
    found_track = False

    for _ in range(40):
        ret, frame = cap.read()
        if not ret:
            break
        tracks = tracker.update(frame)
        if len(tracks) > 0:
            found_track = True
            for trk in tracks:
                # Required fields from Phase 2 specifications
                assert "track_id" in trk and isinstance(trk["track_id"], int)
                assert "class_id" in trk and isinstance(trk["class_id"], int)
                assert "class_name" in trk and isinstance(trk["class_name"], str)
                assert "confidence" in trk and isinstance(trk["confidence"], float)
                assert "bbox" in trk and len(trk["bbox"]) == 4
                assert "center" in trk and len(trk["center"]) == 2
                assert "smoothed_center" in trk and len(trk["smoothed_center"]) == 2
                assert "bottom_center" in trk and len(trk["bottom_center"]) == 2
                assert "trajectory" in trk and isinstance(trk["trajectory"], list)
                assert trk["class_name"] in VehicleDetector.TARGET_VEHICLE_CLASSES.values()
            break

    cap.release()
    assert found_track, "Expected at least one active track within 40 frames"
    print("✓ Track dictionary structure and typing verified.")


def test_tracker_reset() -> None:
    """Verify tracker state resets properly."""
    detector = VehicleDetector(model_path="models/yolo11n.pt")
    tracker = VehicleTracker(tracker_type="bytetrack", detector=detector)

    # Simulate fake trajectory state
    tracker.tracker.trajectories[99] = [ (100, 100), (102, 104) ] # type: ignore
    tracker.tracker.track_hits[99] = 10

    tracker.reset()
    assert len(tracker.tracker.trajectories) == 0
    assert len(tracker.tracker.track_hits) == 0
    print("✓ Tracker reset functionality verified.")


def test_tracker_drawing() -> None:
    """Verify visual track annotation overlay."""
    detector = VehicleDetector(model_path="models/yolo11n.pt")
    tracker = VehicleTracker(tracker_type="bytetrack", detector=detector)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_tracks = [
        {
            "track_id": 1,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.88,
            "bbox": [50.0, 50.0, 150.0, 150.0],
            "center": (100, 100),
            "smoothed_center": (100, 100),
            "bottom_center": (100, 150),
            "trajectory": [(95, 95), (100, 100)],
        }
    ]

    annotated = tracker.draw_tracks(dummy_frame, sample_tracks)
    assert annotated.shape == dummy_frame.shape
    assert annotated.dtype == dummy_frame.dtype
    print("✓ Track rendering verified.")


if __name__ == "__main__":
    print("--- Running VehicleTracker Unit Tests ---")
    test_tracker_initialization()
    test_track_data_structure()
    test_tracker_reset()
    test_tracker_drawing()
    print("--- All Tracker Unit Tests Passed Successfully ---")
