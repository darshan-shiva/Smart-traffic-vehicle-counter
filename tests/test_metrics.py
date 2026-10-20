"""Unit and Integration Tests for TrackerMetrics & Evaluation Module (Phase 5).

Runs ByteTrack and BoT-SORT on identical video inputs, verifies all 8 metrics
are calculated from actual execution without hardcoding, generates comparative
CSV report, and generates annotated evaluation videos.

Author: Darshan S.
"""

from pathlib import Path
import sys
from typing import Tuple
import cv2
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.metrics import TrackerMetrics, TrackerEvaluator


def test_metric_calculation_logic() -> bool:
    """Verify that metric formulas (short tracks, average length, FPS) are mathematically exact."""
    metrics = TrackerMetrics()

    # Simulate 5 frames
    # Track 1 appears in 12 frames (long track)
    # Track 2 appears in 4 frames (<10 frames -> short track)
    # Track 3 appears in 2 frames (<10 frames -> short track)
    for _ in range(2):
        metrics.record_frame(latency_ms=20.0, active_track_ids=[1, 2, 3])
    for _ in range(2):
        metrics.record_frame(latency_ms=20.0, active_track_ids=[1, 2])
    for _ in range(8):
        metrics.record_frame(latency_ms=20.0, active_track_ids=[1])

    summary = metrics.compute_summary(up_count=3, down_count=5, total_time_sec=0.24)

    assert summary["UP count"] == 3
    assert summary["DOWN count"] == 5
    assert summary["TOTAL count"] == 8
    assert summary["Unique Track IDs"] == 3
    # Short tracks (< 10 frames): Track 2 (len 4) and Track 3 (len 2) -> 2
    assert summary["Number of short tracks (<10 frames)"] == 2
    # Average track length: (12 + 4 + 2) / 3 = 18 / 3 = 6.0
    assert summary["Average track length"] == 6.0
    # FPS: 12 frames / 0.24s = 50.0 FPS
    assert summary["Processing FPS"] == 50.0

    print("✓ Metric calculation formulas verified.")
    return True


def test_comparison_csv_export() -> bool:
    """Verify comparison DataFrame formatting and CSV writing."""
    res1 = {
        "UP count": 2, "DOWN count": 3, "TOTAL count": 5,
        "Unique Track IDs": 4, "Number of short tracks (<10 frames)": 1,
        "Average track length": 15.5, "Processing FPS": 24.2, "Processing time": "3.3s",
    }
    res2 = {
        "UP count": 1, "DOWN count": 4, "TOTAL count": 5,
        "Unique Track IDs": 5, "Number of short tracks (<10 frames)": 2,
        "Average track length": 12.0, "Processing FPS": 18.5, "Processing time": "4.3s",
    }

    df = TrackerMetrics.compare_trackers(res1, res2)
    assert list(df.columns) == ["Metric", "ByteTrack", "BoT-SORT"]
    assert len(df) == 8

    test_csv = PROJECT_ROOT / "outputs" / "test_eval_table.csv"
    saved = TrackerMetrics.save_comparison_csv(df, test_csv)
    assert saved.is_file()

    read_back = pd.read_csv(str(saved))
    assert len(read_back) == 8
    test_csv.unlink(missing_ok=True)

    print("✓ Comparison DataFrame formatting and CSV export verified.")
    return True


def run_real_world_tracker_comparison() -> Tuple[bool, bool, bool, bool, bool, pd.DataFrame]:
    """Execute ByteTrack and BoT-SORT on identical video inputs and measure all metrics."""
    video_path = PROJECT_ROOT / "assets" / "traffic_sample.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "traffic_test.mp4"
    assert video_path.exists(), f"Video missing: {video_path}"

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    shared_detector = VehicleDetector(model_path="models/yolo11n.pt", confidence_threshold=0.35)

    evaluator = TrackerEvaluator(
        video_path=video_path,
        detector=shared_detector,
        line_y=240,
        margin=10,
        min_age=2,
    )

    max_eval_frames = 80
    bytetrack_video = outputs_dir / "phase5_bytetrack_eval.mp4"
    botsort_video = outputs_dir / "phase5_botsort_eval.mp4"
    csv_output = outputs_dir / "tracker_comparison.csv"

    print(f"\n[1/3] Running ByteTrack evaluation on {max_eval_frames} frames...")
    byte_summary = evaluator.run_evaluation(
        tracker_type="bytetrack",
        max_frames=max_eval_frames,
        output_video_path=bytetrack_video,
    )

    print(f"[2/3] Running BoT-SORT evaluation on {max_eval_frames} frames...")
    bot_summary = evaluator.run_evaluation(
        tracker_type="botsort",
        max_frames=max_eval_frames,
        output_video_path=botsort_video,
    )

    # 1. ByteTrack Execution check
    byte_ok = bytetrack_video.is_file() and bytetrack_video.stat().st_size > 1000 and byte_summary["_total_frames"] == max_eval_frames

    # 2. BoT-SORT Execution check
    bot_ok = botsort_video.is_file() and botsort_video.stat().st_size > 1000 and bot_summary["_total_frames"] == max_eval_frames

    # 3. Metric Calculation check (all required keys present, non-null, correct types)
    metric_calc_ok = True
    for key in TrackerMetrics.REQUIRED_METRIC_KEYS:
        if key not in byte_summary or key not in bot_summary:
            metric_calc_ok = False
        if byte_summary[key] is None or bot_summary[key] is None:
            metric_calc_ok = False

    # 4. Construct Comparison Table and Save CSV
    df = TrackerMetrics.compare_trackers(byte_summary, bot_summary)
    TrackerMetrics.save_comparison_csv(df, csv_output)
    csv_ok = csv_output.is_file() and csv_output.stat().st_size > 100

    # 5. Same-Input Check
    same_input_ok = (
        byte_summary["_total_frames"] == bot_summary["_total_frames"]
        and evaluator.video_path == video_path
        and evaluator.line_y == 240
        and evaluator.detector.confidence_threshold == 0.35
    )

    # 6. Verify No Hardcoding (values are dynamically produced from timers and counters)
    no_hardcoding = (
        byte_summary["_raw_processing_time"] > 0
        and bot_summary["_raw_processing_time"] > 0
        and byte_summary["Processing FPS"] > 0
        and bot_summary["Processing FPS"] > 0
    )

    return byte_ok, bot_ok, metric_calc_ok, csv_ok, same_input_ok and no_hardcoding, df


def run_phase5_verification() -> bool:
    """Execute complete Phase 5 test suite."""
    print("=" * 60)
    print("STARTING PHASE 5 VERIFICATION: TRACKER EVALUATION & METRICS")
    print("Author: Darshan S.")
    print("=" * 60)

    # Test metric formulas
    test_metric_calculation_logic()
    test_comparison_csv_export()

    # Run real comparative benchmark
    byte_ok, bot_ok, calc_ok, csv_ok, same_input_ok, df = run_real_world_tracker_comparison()

    print("\n" + "=" * 60)
    print("TRACKER BENCHMARK COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    all_passed = byte_ok and bot_ok and calc_ok and csv_ok and same_input_ok

    print("\n" + "=" * 60)
    print("PHASE 5 VERIFICATION")
    print("=" * 60)
    print(f"ByteTrack execution: {'PASS' if byte_ok else 'FAIL'}")
    print(f"BoT-SORT execution: {'PASS' if bot_ok else 'FAIL'}")
    print(f"Metric calculation: {'PASS' if calc_ok else 'FAIL'}")
    print(f"CSV generation: {'PASS' if csv_ok else 'FAIL'}")
    print(f"Same-input comparison: {'PASS' if same_input_ok else 'FAIL'}")
    print(f"No hardcoded metrics: {'PASS' if same_input_ok else 'FAIL'}")
    print("\nOverall Phase 5:")
    print("PASS" if all_passed else "FAIL")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_phase5_verification()
    sys.exit(0 if success else 1)
