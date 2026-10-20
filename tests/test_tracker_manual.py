"""Manual and End-to-End Verification Script for Phase 2: Multi-Object Tracking.

Tests ByteTrack and BoT-SORT on a real traffic surveillance video clip,
validates persistent IDs, multiple simultaneous object tracking, and generates
annotated MP4 video outputs.

Author: Darshan S.
"""

from collections import Counter
from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.tracker import VehicleTracker


def test_tracker_on_video(
    tracker_type: str,
    video_path: Path,
    output_path: Path,
    detector: VehicleDetector,
    max_frames: int = 70,
) -> dict:
    """Run tracking evaluation on video for a specific tracker algorithm.

    Args:
        tracker_type: 'bytetrack' or 'botsort'
        video_path: Input traffic video path
        output_path: Output annotated MP4 path
        detector: Shared VehicleDetector instance
        max_frames: Maximum number of frames to process for the test

    Returns:
        Dictionary of validation results and metrics.
    """
    print(f"\n{'=' * 20} Testing {tracker_type.upper()} {'=' * 20}")
    status = {
        "loading": False,
        "detection_integration": False,
        "ids_assigned": False,
        "id_persistence": False,
        "multi_object": False,
        "video_generated": False,
        "max_simultaneous": 0,
        "unique_track_ids": set(),
    }

    # 1. Tracker Initialization
    try:
        tracker = VehicleTracker(
            tracker_type=tracker_type,
            detector=detector,
            min_hits=1,
            trajectory_history=30,
        )
        status["loading"] = True
        print(f"✓ {tracker_type.upper()} initialized successfully.")
    except Exception as exc:
        print(f"✗ Failed to initialize {tracker_type}: {exc}")
        return status

    # 2. Open Video Capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return status

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3. Setup Video Writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    track_appearances = Counter()  # track_id -> frame count
    max_simultaneous = 0

    print(f"Processing up to {max_frames} frames from {video_path.name}...")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracker update
        tracks = tracker.update(frame)
        active_ids = [t["track_id"] for t in tracks]

        if len(active_ids) > 0:
            status["detection_integration"] = True
            status["ids_assigned"] = True
            for tid in active_ids:
                track_appearances[tid] += 1
                status["unique_track_ids"].add(tid)

        if len(active_ids) > max_simultaneous:
            max_simultaneous = len(active_ids)

        if len(active_ids) > 1:
            status["multi_object"] = True

        # Render visual annotations
        annotated_frame = tracker.draw_tracks(
            frame,
            tracks,
            draw_trails=True,
            draw_centers=True,
            draw_bottom_center=True,
        )

        # Draw HUD info
        hud_text = f"{tracker_type.upper()} | Frame: {frame_idx} | Active Tracks: {len(tracks)} (IDs: {active_ids})"
        cv2.putText(
            annotated_frame,
            hud_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

    status["max_simultaneous"] = max_simultaneous

    # Verify ID persistence: at least one track must be visible across multiple consecutive frames
    persistent_tracks = [tid for tid, count in track_appearances.items() if count >= 5]
    if len(persistent_tracks) > 0:
        status["id_persistence"] = True

    # 4. Verify Output Video
    if output_path.is_file() and output_path.stat().st_size > 1000:
        verify_cap = cv2.VideoCapture(str(output_path))
        written_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        verify_cap.release()
        if written_frames >= frame_idx - 2:
            status["video_generated"] = True

    print(f"Results for {tracker_type.upper()}:")
    print(f"  Processed Frames: {frame_idx}")
    print(f"  Unique Track IDs observed: {sorted(list(status['unique_track_ids']))}")
    print(f"  Tracks persistent for >= 5 frames: {persistent_tracks}")
    print(f"  Peak simultaneous tracked vehicles: {max_simultaneous}")
    print(f"  Output Video: {output_path} ({output_path.stat().st_size} bytes)")

    return status


def run_phase2_verification() -> bool:
    """Execute end-to-end verification for Phase 2: Multi-Object Tracking."""
    print("=" * 60)
    print("STARTING PHASE 2 VERIFICATION: MULTI-OBJECT TRACKING")
    print("Author: Darshan S.")
    print("=" * 60)

    video_path = PROJECT_ROOT / "assets" / "traffic_sample.mp4"
    if not video_path.exists():
        print(f"✗ Test video missing at: {video_path}")
        return False

    # Initialize shared detector to preserve Phase 1 components
    detector = VehicleDetector(model_path="models/yolo11n.pt", confidence_threshold=0.35)
    print(f"✓ Base VehicleDetector loaded on: {detector.device}")

    outputs_dir = PROJECT_ROOT / "outputs"

    # Test 1: ByteTrack
    bytetrack_output = outputs_dir / "phase2_bytetrack_test.mp4"
    byte_results = test_tracker_on_video(
        tracker_type="bytetrack",
        video_path=video_path,
        output_path=bytetrack_output,
        detector=detector,
        max_frames=70,
    )

    # Test 2: BoT-SORT
    botsort_output = outputs_dir / "phase2_botsort_test.mp4"
    bot_results = test_tracker_on_video(
        tracker_type="botsort",
        video_path=video_path,
        output_path=botsort_output,
        detector=detector,
        max_frames=70,
    )

    # Multi-object tracking confirmed if both trackers track multiple simultaneous vehicles
    multi_object_confirmed = byte_results["multi_object"] and bot_results["multi_object"]

    # Final summary check
    all_passed = (
        byte_results["loading"]
        and byte_results["detection_integration"]
        and byte_results["ids_assigned"]
        and byte_results["id_persistence"]
        and byte_results["video_generated"]
        and bot_results["loading"]
        and bot_results["detection_integration"]
        and bot_results["ids_assigned"]
        and bot_results["id_persistence"]
        and bot_results["video_generated"]
        and multi_object_confirmed
    )

    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION")
    print("=" * 60)
    print(f"ByteTrack loading: {'PASS' if byte_results['loading'] else 'FAIL'}")
    print(f"ByteTrack detection integration: {'PASS' if byte_results['detection_integration'] else 'FAIL'}")
    print(f"ByteTrack IDs: {'PASS' if (byte_results['ids_assigned'] and byte_results['id_persistence']) else 'FAIL'}")
    print(f"ByteTrack output video: {'PASS' if byte_results['video_generated'] else 'FAIL'}")
    print()
    print(f"BoT-SORT loading: {'PASS' if bot_results['loading'] else 'FAIL'}")
    print(f"BoT-SORT detection integration: {'PASS' if bot_results['detection_integration'] else 'FAIL'}")
    print(f"BoT-SORT IDs: {'PASS' if (bot_results['ids_assigned'] and bot_results['id_persistence']) else 'FAIL'}")
    print(f"BoT-SORT output video: {'PASS' if bot_results['video_generated'] else 'FAIL'}")
    print()
    print(f"Multiple-object tracking: {'PASS' if multi_object_confirmed else 'FAIL'}")
    print()
    print("Overall Phase 2:")
    print("PASS" if all_passed else "FAIL")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_phase2_verification()
    sys.exit(0 if success else 1)
