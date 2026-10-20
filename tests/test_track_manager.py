"""Unit and Integration Tests for TrackManager Module (Phase 3).

Verifies track history creation, age progression, minimum age filtering,
center smoothing, previous/current position tracking, stale-track eviction,
and generates outputs/phase3_trajectory_test.mp4 on real traffic video.

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
from src.track_manager import TrackManager, TrackedVehicle


def test_track_history_and_age() -> bool:
    """Verify track history creation and age incrementation."""
    tm = TrackManager(min_age=3, max_staleness=5, history_len=20)

    # Frame 1: Register track ID 1
    raw_frame_1 = [
        {"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox": [10, 10, 50, 50]}
    ]
    tracks_f1 = tm.update(raw_frame_1, filter_unconfirmed=False)
    assert len(tracks_f1) == 1
    t1 = tracks_f1[0]
    assert t1.age == 1
    assert t1.hits == 1
    assert t1.previous_center is None
    assert len(t1.history) == 1
    assert t1.is_confirmed is False  # age 1 < min_age 3

    # Frame 2: Update track ID 1
    raw_frame_2 = [
        {"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.92, "bbox": [12, 14, 52, 54]}
    ]
    tracks_f2 = tm.update(raw_frame_2, filter_unconfirmed=False)
    t1_f2 = tracks_f2[0]
    assert t1_f2.age == 2
    assert t1_f2.hits == 2
    assert t1_f2.previous_center is not None
    assert len(t1_f2.history) == 2
    assert t1_f2.is_confirmed is False  # age 2 < min_age 3

    # Frame 3: Update track ID 1 -> meets min_age 3
    raw_frame_3 = [
        {"track_id": 1, "class_id": 2, "class_name": "car", "confidence": 0.91, "bbox": [14, 18, 54, 58]}
    ]
    tracks_f3 = tm.update(raw_frame_3, filter_unconfirmed=True)
    assert len(tracks_f3) == 1
    t1_f3 = tracks_f3[0]
    assert t1_f3.age == 3
    assert t1_f3.is_confirmed is True
    assert len(t1_f3.history) == 3

    print("✓ Track history and age progression verified.")
    return True


def test_track_filtering() -> bool:
    """Verify that transient/short-lived tracks are ignored."""
    tm = TrackManager(min_age=4, max_staleness=5)

    # Track 10 appears for only 2 frames (transient jitter)
    f1 = [{"track_id": 10, "class_id": 2, "class_name": "car", "confidence": 0.8, "bbox": [10, 10, 30, 30]}]
    f2 = [{"track_id": 10, "class_id": 2, "class_name": "car", "confidence": 0.8, "bbox": [12, 12, 32, 32]}]

    out1 = tm.update(f1, filter_unconfirmed=True)
    out2 = tm.update(f2, filter_unconfirmed=True)
    assert len(out1) == 0, "Unconfirmed track should be filtered out on frame 1"
    assert len(out2) == 0, "Unconfirmed track should be filtered out on frame 2"

    # Track 10 disappears on frame 3 -> ignored completely
    f3 = []
    out3 = tm.update(f3, filter_unconfirmed=True)
    assert len(out3) == 0

    print("✓ Short-lived track filtering verified.")
    return True


def test_center_smoothing() -> bool:
    """Verify moving-average center smoothing eliminates bounding box jitter."""
    tm = TrackManager(min_age=1, smoothing_window=3)

    # Simulate noisy detections along a linear path: x = 100, y increases with noise
    raw_points = [
        (100, 100),
        (105, 110),  # jitter in x (+5)
        (95, 120),   # jitter in x (-5)
        (102, 130),
    ]

    smoothed_x = []
    for cx, cy in raw_points:
        raw = [{
            "track_id": 5,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": [cx - 20, cy - 20, cx + 20, cy + 20],
            "center": (cx, cy),
        }]
        tracks = tm.update(raw, filter_unconfirmed=False)
        smoothed_x.append(tracks[0].current_center[0])

    # Check that smoothed coordinates soften extreme jitter
    assert len(smoothed_x) == 4
    # On frame 3 (recent: 100, 105, 95), avg x should be exactly 100
    assert smoothed_x[2] == 100
    print("✓ Center smoothing calculation verified.")
    return True


def test_stale_track_handling() -> bool:
    """Verify stale tracks are evicted when missing for > max_staleness frames."""
    tm = TrackManager(min_age=1, max_staleness=3)

    # Track 7 appears on frame 1
    tm.update([{"track_id": 7, "class_id": 2, "class_name": "car", "confidence": 0.9, "bbox": [10, 10, 40, 40]}])
    assert 7 in tm.active_tracks

    # Frames 2, 3, 4: Track 7 is missing (staleness 1, 2, 3)
    tm.update([])
    assert 7 in tm.active_tracks and tm.active_tracks[7].time_since_update == 1
    tm.update([])
    assert 7 in tm.active_tracks and tm.active_tracks[7].time_since_update == 2
    tm.update([])
    assert 7 in tm.active_tracks and tm.active_tracks[7].time_since_update == 3

    # Frame 5: Track 7 missing again (staleness 4 > max_staleness 3) -> should be evicted
    tm.update([])
    assert 7 not in tm.active_tracks, "Track 7 should have been pruned as stale"
    assert tm.total_stale_pruned == 1

    print("✓ Stale track eviction verified.")
    return True


def test_video_trajectory_integration() -> Tuple[bool, bool]:
    """Run TrackManager with YOLO and Tracker on real traffic video and generate output video.

    Returns:
        Tuple of (trajectory_tracking_passed, video_generated_passed).
    """
    video_path = PROJECT_ROOT / "assets" / "traffic_sample.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "traffic_test.mp4"
    assert video_path.exists(), f"Test traffic video missing at {video_path}"

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase3_trajectory_test.mp4"

    # Setup detector, tracker, and track manager
    detector = VehicleDetector(model_path="models/yolo11n.pt", confidence_threshold=0.35)
    tracker = VehicleTracker(tracker_type="bytetrack", detector=detector, min_hits=1)
    track_manager = TrackManager(min_age=2, max_staleness=10, history_len=30, smoothing_window=5)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Could not open traffic video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    max_frames = 80
    trajectories_verified = False

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Tracker produces raw detections with IDs
        raw_tracks = tracker.update(frame)

        # 2. TrackManager stabilizes tracks, updates age, history, and smoothing
        managed_tracks = track_manager.update(raw_tracks, filter_unconfirmed=True)

        # Verify trajectory properties when tracks are active
        for trk in managed_tracks:
            if trk.age > 1:
                assert trk.previous_center is not None
                assert len(trk.history) >= 2
                trajectories_verified = True

        # 3. Draw managed tracks (boxes, IDs, age, smoothed centers, motion vectors, trails)
        annotated_frame = track_manager.draw_managed_tracks(
            frame,
            managed_tracks,
            draw_boxes=True,
            draw_trails=True,
            draw_centers=True,
            draw_vectors=True,
        )

        # Draw HUD info
        active_ids = [t.track_id for t in managed_tracks]
        hud_text = f"Phase 3: Trajectory & TrackManager | Frame: {frame_idx} | Confirmed Tracks: {len(managed_tracks)} (IDs: {active_ids})"
        cv2.putText(
            annotated_frame,
            hud_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Verify generated video integrity
    video_generated = False
    if output_path.is_file() and output_path.stat().st_size > 1000:
        verify_cap = cv2.VideoCapture(str(output_path))
        written_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        verify_cap.release()
        if written_count >= frame_idx - 2:
            video_generated = True

    print(f"✓ Video trajectory test completed: {frame_idx} frames written to {output_path} ({output_path.stat().st_size:,} bytes)")
    return trajectories_verified, video_generated


def run_phase3_verification() -> bool:
    """Execute complete Phase 3 verification suite."""
    print("=" * 60)
    print("STARTING PHASE 3 VERIFICATION: TRACK MANAGEMENT & TRAJECTORIES")
    print("Author: Darshan S.")
    print("=" * 60)

    results = {}

    try:
        results["Track history"] = test_track_history_and_age()
        results["Track age"] = results["Track history"]
    except Exception as exc:
        print(f"✗ Track history / age test failed: {exc}")
        results["Track history"] = False
        results["Track age"] = False

    try:
        results["Track filtering"] = test_track_filtering()
    except Exception as exc:
        print(f"✗ Track filtering test failed: {exc}")
        results["Track filtering"] = False

    try:
        results["Center smoothing"] = test_center_smoothing()
    except Exception as exc:
        print(f"✗ Center smoothing test failed: {exc}")
        results["Center smoothing"] = False

    try:
        results["Stale-track handling"] = test_stale_track_handling()
    except Exception as exc:
        print(f"✗ Stale track handling test failed: {exc}")
        results["Stale-track handling"] = False

    try:
        traj_ok, vid_ok = test_video_trajectory_integration()
        results["Trajectory tracking"] = traj_ok
        results["Output video"] = vid_ok
    except Exception as exc:
        print(f"✗ Video trajectory integration failed: {exc}")
        results["Trajectory tracking"] = False
        results["Output video"] = False

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION")
    print("=" * 60)
    for key, passed in results.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")
    print("\nOverall Phase 3:")
    print("PASS" if all_passed else "FAIL")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_phase3_verification()
    sys.exit(0 if success else 1)
