"""Unit and Integration Tests for VehicleCounter Module (Phase 4).

Tests configurable counting line, neutral margin zone, bidirectional UP/DOWN
crossing logic, duplicate counting prevention, total count arithmetic,
and generates outputs/phase4_counting_test.mp4 with HUD annotations on real video.

Author: Darshan S.
"""

from pathlib import Path
import sys
from typing import Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.track_manager import TrackManager
from src.counter import VehicleCounter


def test_counting_line_configuration() -> bool:
    """Verify line position configuration and height resolution."""
    # 1. Explicit line position
    c1 = VehicleCounter(line_position_y=300)
    assert c1.line_position_y == 300

    # 2. Ratio based on frame height
    c2 = VehicleCounter(line_ratio=0.5)
    resolved_y = c2.resolve_line_y(frame_height=600)
    assert resolved_y == 300
    assert c2.line_position_y == 300

    # 3. Dynamic setter
    c1.line_position_y = 350
    assert c1.line_position_y == 350

    print("✓ Counting line configuration verified.")
    return True


def test_bidirectional_and_duplicate_prevention() -> bool:
    """Verify UP crossing, DOWN crossing, and strict duplicate prevention."""
    counter = VehicleCounter(line_position_y=250, margin=10)

    # 1. Test DOWN crossing: Track 1 moves from Y=200 to Y=300
    # First frame (origin)
    counter.check_crossing(track_id=1, class_name="car", current_center=(100, 200), previous_center=None)
    assert counter.total_count == 0
    # Second frame (crosses down to 300)
    dir_1 = counter.check_crossing(track_id=1, class_name="car", current_center=(100, 300), previous_center=(100, 200))
    assert dir_1 == "DOWN"
    assert counter.down_count == 1
    assert counter.up_count == 0
    assert counter.total_count == 1

    # Duplicate test: Track 1 continues moving or wobbles back up across line
    dup_dir = counter.check_crossing(track_id=1, class_name="car", current_center=(100, 210), previous_center=(100, 300))
    assert dup_dir is None, "Track 1 was already counted and must never be counted again"
    assert counter.down_count == 1
    assert counter.total_count == 1

    # 2. Test UP crossing: Track 2 moves from Y=320 to Y=180
    counter.check_crossing(track_id=2, class_name="bus", current_center=(200, 320), previous_center=None)
    dir_2 = counter.check_crossing(track_id=2, class_name="bus", current_center=(200, 180), previous_center=(200, 320))
    assert dir_2 == "UP"
    assert counter.up_count == 1
    assert counter.down_count == 1
    assert counter.total_count == 2
    assert counter.class_counts["bus"]["up"] == 1
    assert counter.class_counts["car"]["down"] == 1

    # Duplicate test for Track 2
    dup_dir2 = counter.check_crossing(track_id=2, class_name="bus", current_center=(200, 160), previous_center=(200, 180))
    assert dup_dir2 is None
    assert counter.up_count == 1
    assert counter.total_count == 2

    print("✓ UP/DOWN detection and duplicate prevention verified.")
    return True


def test_neutral_zone_and_non_crossing() -> bool:
    """Verify neutral margin zone prevents jitter and non-crossing vehicles are not counted."""
    counter = VehicleCounter(line_position_y=250, margin=10)

    # 1. Non-crossing vehicle: stays far above line (Y in [80, 120])
    for y in range(80, 130, 10):
        counter.check_crossing(track_id=99, class_name="car", current_center=(50, y), previous_center=(50, y - 10))
    assert 99 not in counter.counted_ids
    assert counter.total_count == 0

    # 2. Jitter within neutral zone: track 50 wobbles between 245 and 255 (margin is ±10, i.e., [240, 260])
    counter.check_crossing(track_id=50, class_name="truck", current_center=(100, 245), previous_center=None)
    counter.check_crossing(track_id=50, class_name="truck", current_center=(100, 255), previous_center=(100, 245))
    counter.check_crossing(track_id=50, class_name="truck", current_center=(100, 248), previous_center=(100, 255))
    assert 50 not in counter.counted_ids
    assert counter.total_count == 0

    print("✓ Neutral margin zone and non-crossing vehicles verified.")
    return True


def test_total_calculation() -> bool:
    """Verify total_count strictly equals up_count + down_count."""
    counter = VehicleCounter(line_position_y=200)
    assert counter.total_count == 0

    counter.up_count = 7
    counter.down_count = 13
    assert counter.total_count == 20
    assert counter.total_count == (counter.up_count + counter.down_count)

    print("✓ Total count arithmetic verified.")
    return True


def test_video_counting_integration() -> bool:
    """Run full pipeline on traffic video and generate phase4_counting_test.mp4."""
    video_path = PROJECT_ROOT / "assets" / "traffic_sample.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "traffic_test.mp4"
    assert video_path.exists(), f"Video missing: {video_path}"

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase4_counting_test.mp4"

    # Pipeline setup
    detector = VehicleDetector(model_path="models/yolo11n.pt", confidence_threshold=0.35)
    tracker = VehicleTracker(tracker_type="bytetrack", detector=detector, min_hits=1)
    track_manager = TrackManager(min_age=2, max_staleness=12, history_len=30, smoothing_window=5)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Place line where traffic flows (around Y = 240 in 540p video)
    counter = VehicleCounter(line_position_y=240, margin=10)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    max_frames = 100

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detection + Tracking
        raw_tracks = tracker.update(frame)

        # 2. Track Lifecycle & Trajectory Processing
        managed_tracks = track_manager.update(raw_tracks, filter_unconfirmed=True)

        # 3. Line-Crossing Vehicle Counting
        crossing_events = counter.update(managed_tracks)

        # 4. Render Tracks and Trajectories (bounding boxes, IDs, centers, trails)
        annotated_frame = track_manager.draw_managed_tracks(
            frame,
            managed_tracks,
            draw_boxes=True,
            draw_trails=True,
            draw_centers=True,
            draw_vectors=True,
        )

        # 5. Render Counting Line, Neutral Zone, and HUD Dashboard (UP, DOWN, TOTAL)
        annotated_frame = counter.draw_counter(annotated_frame, managed_tracks)

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Verify generated output video
    assert output_path.is_file(), f"Output file missing: {output_path}"
    assert output_path.stat().st_size > 1000, "Output video too small"

    verify_cap = cv2.VideoCapture(str(output_path))
    written_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    verify_cap.release()
    assert written_frames >= frame_idx - 2

    print(f"✓ Video counting integration test completed: {written_frames} frames generated at {output_path}")
    print(f"  Final Counts -> UP: {counter.up_count} | DOWN: {counter.down_count} | TOTAL: {counter.total_count}")
    return True


def run_phase4_verification() -> bool:
    """Run all Phase 4 verification tests and report results."""
    print("=" * 60)
    print("STARTING PHASE 4 VERIFICATION: LINE-CROSSING VEHICLE COUNTING")
    print("Author: Darshan S.")
    print("=" * 60)

    results = {}

    try:
        results["Counting line"] = test_counting_line_configuration()
    except Exception as exc:
        print(f"✗ Counting line test failed: {exc}")
        results["Counting line"] = False

    try:
        bi_ok = test_bidirectional_and_duplicate_prevention()
        results["UP detection"] = bi_ok
        results["DOWN detection"] = bi_ok
        results["Duplicate prevention"] = bi_ok
    except Exception as exc:
        print(f"✗ Crossing detection test failed: {exc}")
        results["UP detection"] = False
        results["DOWN detection"] = False
        results["Duplicate prevention"] = False

    try:
        results["Neutral zone"] = test_neutral_zone_and_non_crossing()
    except Exception as exc:
        print(f"✗ Neutral zone test failed: {exc}")
        results["Neutral zone"] = False

    try:
        results["Total calculation"] = test_total_calculation()
    except Exception as exc:
        print(f"✗ Total calculation test failed: {exc}")
        results["Total calculation"] = False

    try:
        results["Output video"] = test_video_counting_integration()
    except Exception as exc:
        print(f"✗ Video integration test failed: {exc}")
        results["Output video"] = False

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    print("PHASE 4 VERIFICATION")
    print("=" * 60)
    for key, passed in results.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")
    print("\nOverall Phase 4:")
    print("PASS" if all_passed else "FAIL")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_phase4_verification()
    sys.exit(0 if success else 1)
