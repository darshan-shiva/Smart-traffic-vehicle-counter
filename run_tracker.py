"""Run YOLO11 Vehicle Detection & Tracking on a Video.

Supports ByteTrack and BoT-SORT trackers with persistent track IDs,
bounding boxes, class labels, confidence scores, and trajectory trails.

Author: Darshan S.

Examples:
    # Run ByteTrack on sample video
    python run_tracker.py --tracker bytetrack --video assets/traffic_sample.mp4 --output outputs/bytetrack_demo.mp4

    # Run BoT-SORT on sample video
    python run_tracker.py --tracker botsort --video assets/traffic_sample.mp4 --output outputs/botsort_demo.mp4
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.tracker import VehicleTracker


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Traffic Vehicle Tracker (ByteTrack & BoT-SORT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Tracking algorithm to use ('bytetrack' or 'botsort')",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/traffic_sample.mp4",
        help="Path to input traffic video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for saving annotated output video (.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="Path to YOLO11 model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Max frames to process (0 or negative processes the entire video)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=1,
        help="Minimum consecutive detections required before track is displayed",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video window live (press 'q' to stop)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute vehicle detection and tracking."""
    args = parse_arguments()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Input video '{video_path}' does not exist.")
        sys.exit(1)

    if args.output is None:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.tracker}_demo.mp4"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SMART TRAFFIC VEHICLE TRACKER")
    print("Author: Darshan S.")
    print("=" * 65)
    print(f"Tracker Algorithm  : {args.tracker.upper()}")
    print(f"Input Video        : {video_path}")
    print(f"Output Video       : {output_path}")
    print(f"Confidence Thresh  : {args.conf}")
    print(f"Max Frames         : {'All' if args.frames <= 0 else args.frames}")
    print("=" * 65)

    # 1. Initialize Detector and Tracker
    print("\n[1/3] Loading YOLO11 model and initializing tracker...")
    detector = VehicleDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
    )
    tracker = VehicleTracker(
        tracker_type=args.tracker,
        detector=detector,
        min_hits=args.min_hits,
        trajectory_history=30,
    )
    print(f"✓ Detector compute device : {detector.device.upper()}")
    print(f"✓ Configured tracker      : {args.tracker.upper()}")

    # 2. Open Video Stream
    print("\n[2/3] Opening input video...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Video properties: {width}x{height} @ {fps:.1f} FPS | Total frames: {total_video_frames}")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    target_frames = total_video_frames if args.frames <= 0 else min(args.frames, total_video_frames)
    frame_count = 0
    unique_track_ids = set()
    start_time = time.time()

    print(f"\n[3/3] Processing video frames (target: {target_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run multi-object tracker
        tracks = tracker.update(frame)

        # Collect unique IDs
        for trk in tracks:
            unique_track_ids.add(trk["track_id"])

        # Annotate bounding boxes, IDs, labels, motion trails, center dots
        annotated_frame = tracker.draw_tracks(
            frame,
            tracks,
            draw_trails=True,
            draw_centers=True,
            draw_bottom_center=True,
        )

        # Draw HUD banner on the top-left
        hud_line1 = f"System: Smart Traffic Vehicle Counter | Author: Darshan S."
        hud_line2 = f"Tracker: {args.tracker.upper()} | Frame: {frame_count}/{target_frames} | Active: {len(tracks)}"
        active_ids_str = f"Active IDs: {[t['track_id'] for t in tracks]}"

        # Semi-transparent HUD background
        cv2.rectangle(annotated_frame, (10, 10), (width - 10, 85), (20, 20, 20), -1)
        cv2.putText(annotated_frame, hud_line1, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, hud_line2, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, active_ids_str, (20, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA)

        out.write(annotated_frame)

        # Print progress every 15 frames
        if frame_count % 15 == 0 or frame_count == target_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_count:3d}/{target_frames} | Active Tracks: {len(tracks):2d} | Total Unique IDs: {len(unique_track_ids):2d} | Processing Speed: {proc_fps:.1f} FPS")

        if args.show:
            cv2.imshow("Smart Traffic Vehicle Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrupted by user.")
                break

        if args.frames > 0 and frame_count >= args.frames:
            break

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 65)
    print("TRACKING COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Algorithm Used     : {args.tracker.upper()}")
    print(f"Frames Processed   : {frame_count}")
    print(f"Average Speed      : {avg_fps:.1f} FPS")
    print(f"Total Unique IDs   : {len(unique_track_ids)} (IDs: {sorted(list(unique_track_ids))})")
    print(f"Output Video File  : {output_path.resolve()}")
    print(f"File Size          : {output_path.stat().st_size:,} bytes")
    print("=" * 65)


if __name__ == "__main__":
    main()
