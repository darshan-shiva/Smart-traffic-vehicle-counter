"""
Smart Traffic Vehicle Counter
Phase 6 - End-to-End Pipeline

Pipeline:
YOLO11 Detection
    ↓
ByteTrack / BoT-SORT
    ↓
Track Management
    ↓
Trajectory Smoothing
    ↓
Line-Crossing Counting
    ↓
Performance Metrics
    ↓
Annotated Output Video

Author: Darshan S.
"""
from networkx.algorithms.approximation import treewidth
from src.metrics import TrackerMetrics
from botocore import parsers
from pathlib import Path
import argparse
import json
import sys
import time
from src.speed_estimator import SpeedEstimator

import cv2

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.track_manager import TrackManager
from src.counter import VehicleCounter
from src.metrics import TrackerMetrics


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Smart Traffic Vehicle Counter - End-to-End Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/traffic_test.mp4",
        help="Input traffic video",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output annotated video",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        choices=["bytetrack", "botsort"],
        default="bytetrack",
        help="Tracking algorithm",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="YOLO11 model weights",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process. 0 = entire video",
    )

    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.65,
        help="Counting line position as fraction of frame height",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Neutral zone around counting line",
    )

    parser.add_argument(
        "--min-age",
        type=int,
        default=2,
        help="Minimum track age before confirmation",
    )
    parser.add_argument(
        "--speed-line-1",
        type=float,
        default=0.56,
        help="First speed measurement line as frame-height ratio",
    )

    parser.add_argument(
        "--speed-line-2",
        type=float,
        default=0.74,
        help="Second speed measurement line as frame-height ratio",
    )

    parser.add_argument(
        "--speed-distance",
        type=float,
        default=10.0,
        help="Real-world distance between speed lines in meters",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:

    args = parse_arguments()

    input_path = Path(args.input)

    if not input_path.is_file():
        print(f"ERROR: Input video not found: {input_path}")
        sys.exit(1)

    # -----------------------------------------------------
    # Output path
    # -----------------------------------------------------

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            output_dir /
            f"phase6_{args.tracker}.mp4"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print("=" * 70)
    print("SMART TRAFFIC VEHICLE COUNTER")
    print("PHASE 6 - END-TO-END PIPELINE")
    print("=" * 70)

    print(f"Input Video       : {input_path}")
    print(f"Tracker            : {args.tracker.upper()}")
    print(f"YOLO Model         : {args.model}")
    print(f"Confidence         : {args.conf}")
    print(f"Output             : {output_path}")
    print("=" * 70)

    # -----------------------------------------------------
    # Open video first
    # -----------------------------------------------------

    print("\n[1/6] Opening video...")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    # -----------------------------------------------------
# Speed estimation configuration
# -----------------------------------------------------

    speed_line_1_y = int(height * args.speed_line_1)
    speed_line_2_y = int(height * args.speed_line_2)

    speed_estimator = SpeedEstimator(
        line1_y=speed_line_1_y,
        line2_y=speed_line_2_y,
        distance_meters=args.speed_distance,
        fps=fps,
    )

    # Previous center Y position for each tracked vehicle
    previous_centers = {}

    print(
        f"[SPEED] Line 1 Y = {speed_line_1_y}"
    )

    print(
        f"[SPEED] Line 2 Y = {speed_line_2_y}"
    )

    print(
        f"[SPEED] Distance = {args.speed_distance} m"
    )

    print(
        f"[OK] Video: {width}x{height} | "
        f"{fps:.2f} FPS | "
        f"{total_video_frames} frames"
    )

    # -----------------------------------------------------
    # Determine number of frames
    # -----------------------------------------------------

    if args.max_frames <= 0:
        target_frames = total_video_frames
    else:
        target_frames = min(
            args.max_frames,
            total_video_frames
        )

    # -----------------------------------------------------
    # Initialize detector
    # -----------------------------------------------------

    print("\n[2/6] Loading YOLO11 detector...")

    detector = VehicleDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )

    print(
        f"[OK] YOLO11 loaded "
        f"({detector.device.upper()})"
    )

    # -----------------------------------------------------
    # Initialize tracker
    # -----------------------------------------------------

    print(
        f"\n[3/6] Initializing "
        f"{args.tracker.upper()}..."
    )

    tracker = VehicleTracker(
        tracker_type=args.tracker,
        detector=detector,
        min_hits=1,
        trajectory_history=30,
    )

    print(
        f"[OK]Tracker initialized: "
        f"{args.tracker.upper()}"
    )

    # -----------------------------------------------------
    # Initialize Track Manager
    # -----------------------------------------------------

    track_manager = TrackManager(
        min_age=args.min_age,
        max_staleness=12,
        history_len=30,
        smoothing_window=5,
    )

    print("[OK] Track manager initialized")

    # -----------------------------------------------------
    # Initialize Counter
    # -----------------------------------------------------

    line_y = int(
        height * args.line_ratio
    )

    counter = VehicleCounter(
        line_position_y=line_y,
        margin=args.margin,
    )

    print(
        f"[OK]Counting line: Y={line_y}"
    )

    # -----------------------------------------------------
    # Initialize Metrics
    # -----------------------------------------------------

    metrics = TrackerMetrics()

    # -----------------------------------------------------
    # Video writer
    # -----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    # -----------------------------------------------------
    # Start processing
    # -----------------------------------------------------

    print(
        f"\n[4/6] Processing "
        f"{target_frames} frames..."
    )

    frame_count = 0

    metrics.start_timing()

    pipeline_start = time.time()

    while frame_count < target_frames:

        frame_start = time.time()

        ret, frame = cap.read()

        if not ret:
            break

        # =================================================
        # STEP A — Detection + Tracking
        # =================================================

        raw_tracks = tracker.update(frame)

        # =================================================
        # STEP B — Track Management
        # =================================================

        managed_tracks = track_manager.update(
            raw_tracks,
            filter_unconfirmed=True,
        )
        for track in managed_tracks:

            track_id = track.track_id

            current_y = track.bottom_point[1]

            previous_y = previous_centers.get(
                track_id,
                current_y
            )

            speed_record = speed_estimator.update_track(
                track_id=track_id,
                previous_y=previous_y,
                current_y=current_y,
                frame_number=frame_count,
        
            )

            previous_centers[track_id] = current_y

            if speed_record is not None:
                print(
                    f"[SPEED] Track {track_id} "
                    f"({track.class_name}) -> "
                    f"{speed_record.speed_kmh:.2f} km/h"
                )

        # =================================================
        # STEP C — Vehicle Counting
        # =================================================

        crossing_events = counter.update(
            managed_tracks
        )

        # =================================================
        # STEP D — Metrics
        # =================================================

        frame_latency_ms = (
            time.time() - frame_start
        ) * 1000.0

        active_ids = [
            track.track_id
            for track in managed_tracks
        ]

        metrics.record_frame(
            frame_latency_ms,
            active_ids,
        )

        # =================================================
        # STEP E — Visualization
        # =================================================

        annotated = track_manager.draw_managed_tracks(
            frame,
            managed_tracks,
            draw_boxes=True,
            draw_trails=True,
            draw_centers=True,
            draw_vectors=True,
        )

        # Draw counting line + counts

        annotated = counter.draw_counter(
            annotated,
            managed_tracks,
        )

        # =================================================
        # STEP F — Phase 6 HUD
        # =================================================

        hud_height = 110

        cv2.rectangle(
            annotated,
            (10, 10),
            (width - 10, hud_height),
            (20, 20, 20),
            -1,
        )
        cv2.line(
            annotated,
            (0, speed_line_1_y),
            (width, speed_line_1_y),
            (0, 255, 255),
            2,
        )

        cv2.line(
            annotated,
            (0, speed_line_2_y),
            (width, speed_line_2_y),
            (0, 255, 255),
            2,
    )

        cv2.putText(
            annotated,
            "SMART TRAFFIC VEHICLE COUNTER",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            f"Tracker: {args.tracker.upper()}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "SPEED LINE 1",
            (20, speed_line_1_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            "SPEED LINE 2",
            (20, speed_line_2_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            (
                f"Frame: {frame_count + 1}/"
                f"{target_frames}"
                f" | Active: {len(managed_tracks)}"
            ),
            (20, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"UP: {counter.up_count} | "
                f"DOWN: {counter.down_count} | "
                f"TOTAL: {counter.total_count}"
            ),
            (20, 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        # -------------------------------------------------
        # Display crossing events
        # -------------------------------------------------

        if crossing_events:

            event = crossing_events[-1]

            event_text = (
                f"{event['class_name'].upper()} "
                f"ID {event['track_id']} "
                f"-> {event['direction']}"
            )

            cv2.putText(
                annotated,
                event_text,
                (width - 350, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------
        # Write frame
        # -------------------------------------------------

        writer.write(annotated)

        frame_count += 1

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            frame_count % 15 == 0
            or frame_count == target_frames
        ):

            elapsed = (
                time.time() - pipeline_start
            )

            processing_fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"Frame {frame_count:4d}/"
                f"{target_frames} | "
                f"Active: {len(managed_tracks):2d} | "
                f"UP: {counter.up_count:2d} | "
                f"DOWN: {counter.down_count:2d} | "
                f"TOTAL: {counter.total_count:2d} | "
                f"FPS: {processing_fps:.2f}"
            )

    # -----------------------------------------------------
    # Finish
    # -----------------------------------------------------

    metrics.stop_timing()

    cap.release()
    writer.release()

    # -----------------------------------------------------
    # Calculate final metrics
    # -----------------------------------------------------

    summary = metrics.compute_summary(
        up_count=counter.up_count,
        down_count=counter.down_count,
        class_counts=counter.class_counts,
        speed_summary=speed_estimator.get_summary(),
    )
    

    # -----------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------

    metrics_path = (
        output_dir /
        f"phase6_{args.tracker}_metrics.json"
    )

    clean_summary = {}

    for key, value in summary.items():

        if not key.startswith("_"):
            clean_summary[key] = value

    clean_summary["tracker"] = args.tracker
    clean_summary["input_video"] = str(input_path)
    clean_summary["output_video"] = str(
        output_path
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean_summary,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 6 PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Tracker              : "
        f"{args.tracker.upper()}"
    )

    print(
        f"Frames Processed     : "
        f"{frame_count}"
    )

    print(
        f"UP Count             : "
        f"{counter.up_count}"
    )

    print(
        f"DOWN Count           : "
        f"{counter.down_count}"
    )

    print(
        f"TOTAL Count          : "
        f"{counter.total_count}"
    )

    print(
        f"Unique Track IDs     : "
        f"{summary['Unique Track IDs']}"
    )

    print(
        f"Average Track Length : "
        f"{summary['Average track length']}"
    )

    print(
        f"Processing FPS       : "
        f"{summary['Processing FPS']}"
    )

    print(
        f"Processing Time      : "
        f"{summary['Processing time']}"
    )

    print(
        f"\nOutput Video         : "
        f"{output_path.resolve()}"
    )

    print(
        f"Metrics JSON         : "
        f"{metrics_path.resolve()}"
    )

    print("=" * 70)
    print("PHASE 6 STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()