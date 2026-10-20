"""Tracking Metrics and Performance Evaluation Module.

Benchmarks and compares ByteTrack and BoT-SORT trackers on identical video inputs.
Calculates actual runtime metrics: UP/DOWN/TOTAL counts, unique IDs, short-track count,
average track length, FPS, and execution latency.

Author: Darshan S.
"""

from collections import Counter
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.track_manager import TrackManager
from src.counter import VehicleCounter


class TrackerMetrics:
    """Computes and formats tracking and counting benchmark statistics."""

    REQUIRED_METRIC_KEYS = [
        "UP count",
        "DOWN count",
        "TOTAL count",
        "Unique Track IDs",
        "Number of short tracks (<10 frames)",
        "Average track length",
        "Processing FPS",
        "Processing time",
        "Vehicle Class Counts",
        "Vehicle Class Direction Counts",
    ]

    def __init__(self) -> None:
        """Initialize metrics accumulator."""
        self.frame_latencies_ms: List[float] = []
        self.track_frame_counts: Counter = Counter()  # track_id -> total frames detected
        self.total_frames_processed: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start_timing(self) -> None:
        """Record batch processing start time."""
        self.start_time = time.time()

    def stop_timing(self) -> None:
        """Record batch processing completion time."""
        self.end_time = time.time()

    def record_frame(self, latency_ms: float, active_track_ids: List[int]) -> None:
        """Record latency and active track IDs for a single frame.

        Args:
            latency_ms: Processing latency of the frame in milliseconds.
            active_track_ids: List of active track IDs in this frame.
        """
        self.frame_latencies_ms.append(float(latency_ms))
        self.total_frames_processed += 1
        for tid in active_track_ids:
            self.track_frame_counts[tid] += 1

    def compute_summary(
        self,
        up_count: int,
        down_count: int,
        total_time_sec: Optional[float] = None,
        class_counts: Optional[Dict[str, Dict[str, int]]] = None,
        speed_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate the comprehensive metrics dictionary from actual execution records.

        Args:
            up_count: Measured vehicles crossing UP.
            down_count: Measured vehicles crossing DOWN.
            total_time_sec: Optional total elapsed time in seconds.
            class_counts: Vehicle class counts recorded at line crossings.

        Returns:
            Dictionary containing all required metrics.
        """
        if total_time_sec is None:
            if self.start_time is not None and self.end_time is not None:
                total_time_sec = self.end_time - self.start_time
            elif len(self.frame_latencies_ms) > 0:
                total_time_sec = sum(self.frame_latencies_ms) / 1000.0
            else:
                total_time_sec = 0.0

        total_frames = self.total_frames_processed
        fps = (total_frames / total_time_sec) if total_time_sec > 0 else 0.0

        unique_ids = len(self.track_frame_counts)
        lifespans = list(self.track_frame_counts.values())

        short_tracks = sum(1 for length in lifespans if length < 10)
        avg_track_length = (sum(lifespans) / unique_ids) if unique_ids > 0 else 0.0

        # These are actual counting-line crossing events recorded by VehicleCounter.
        direction_counts = class_counts or {}
        vehicle_class_counts: Dict[str, int] = {}

        for class_name, directions in direction_counts.items():
            if not isinstance(directions, dict):
                continue
            try:
                up = int(directions.get("up", 0))
                down = int(directions.get("down", 0))
            except (TypeError, ValueError):
                up, down = 0, 0
            vehicle_class_counts[str(class_name)] = up + down

        summary = {
            "UP count": int(up_count),
            "DOWN count": int(down_count),
            "TOTAL count": int(up_count + down_count),
            "Unique Track IDs": int(unique_ids),
            "Number of short tracks (<10 frames)": int(short_tracks),
            "Average track length": round(float(avg_track_length), 2),
            "Processing FPS": round(float(fps), 2),
            "Processing time": f"{round(float(total_time_sec), 2)}s",
            "Vehicle Class Counts": vehicle_class_counts,
            "Vehicle Class Direction Counts": direction_counts,
            "_raw_processing_time": float(total_time_sec),
            "_total_frames": int(total_frames),
        }

        # Add speed metrics when available
        if speed_summary:
            summary.update(speed_summary)

        return summary

    @staticmethod
    def compare_trackers(
        bytetrack_results: Dict[str, Any],
        botsort_results: Dict[str, Any],
    ) -> pd.DataFrame:
        """Construct a side-by-side comparative pandas DataFrame.

        Args:
            bytetrack_results: Metrics dictionary from ByteTrack execution.
            botsort_results: Metrics dictionary from BoT-SORT execution.

        Returns:
            Formatted pandas DataFrame with 'Metric', 'ByteTrack', 'BoT-SORT' columns.
        """
        metrics_order = [
            "UP count",
            "DOWN count",
            "TOTAL count",
            "Unique Track IDs",
            "Number of short tracks (<10 frames)",
            "Average track length",
            "Processing FPS",
            "Processing time",
        ]

        rows = []
        for m in metrics_order:
            val_byte = bytetrack_results.get(m, "N/A")
            val_bot = botsort_results.get(m, "N/A")
            rows.append({
                "Metric": m,
                "ByteTrack": val_byte,
                "BoT-SORT": val_bot,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def save_comparison_csv(
        df: pd.DataFrame,
        output_path: Union[str, Path] = "outputs/tracker_comparison.csv",
    ) -> Path:
        """Export comparison table to CSV.

        Args:
            df: DataFrame generated by compare_trackers().
            output_path: Destination CSV file path.

        Returns:
            Resolved Path of the saved CSV.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(p), index=False)
        return p


class TrackerEvaluator:
    """Executes identical video evaluation runs across different trackers."""

    def __init__(
        self,
        video_path: Union[str, Path],
        detector: Optional[VehicleDetector] = None,
        line_y: int = 240,
        margin: int = 10,
        min_age: int = 2,
    ) -> None:
        """Initialize tracker evaluator.

        Args:
            video_path: Input video file path.
            detector: Shared VehicleDetector instance for identical inference.
            line_y: Counting line vertical position.
            margin: Neutral zone margin in pixels.
            min_age: Minimum track age before confirmation.
        """
        self.video_path = Path(video_path)
        self.detector = detector if detector is not None else VehicleDetector(confidence_threshold=0.35)
        self.line_y = line_y
        self.margin = margin
        self.min_age = min_age

    def run_evaluation(
        self,
        tracker_type: str,
        max_frames: int = 80,
        output_video_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Execute a full detection, tracking, management, and counting pipeline run.

        Args:
            tracker_type: 'bytetrack' or 'botsort'.
            max_frames: Total frames to process.
            output_video_path: Optional destination path for annotated output video.

        Returns:
            Evaluated metrics dictionary.
        """
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Evaluation video not found: {self.video_path}")

        tracker = VehicleTracker(
            tracker_type=tracker_type,
            detector=self.detector,
            min_hits=1,
            trajectory_history=30,
        )
        track_manager = TrackManager(
            min_age=self.min_age,
            max_staleness=12,
            history_len=30,
            smoothing_window=5,
        )
        counter = VehicleCounter(line_position_y=self.line_y, margin=self.margin)
        metrics = TrackerMetrics()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_writer = None
        if output_video_path is not None:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_count = 0
        metrics.start_timing()

        while frame_count < max_frames:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Tracking & lifecycle pipeline
            raw_tracks = tracker.update(frame)
            managed_tracks = track_manager.update(raw_tracks, filter_unconfirmed=True)
            counter.update(managed_tracks)

            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            # Record active IDs
            active_ids = [t.track_id for t in managed_tracks]
            metrics.record_frame(latency_ms, active_ids)

            # Render output video if requested
            if out_writer is not None:
                annotated = track_manager.draw_managed_tracks(
                    frame,
                    managed_tracks,
                    draw_boxes=True,
                    draw_trails=True,
                    draw_centers=True,
                    draw_vectors=True,
                )
                annotated = counter.draw_counter(annotated, managed_tracks)

                # Add tracker evaluation watermark
                hud = f"Eval: {tracker_type.upper()} | Frame: {frame_count + 1}/{max_frames} | Active: {len(managed_tracks)}"
                cv2.putText(annotated, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                out_writer.write(annotated)

            frame_count += 1

        metrics.stop_timing()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        print("\n========== FINAL COUNTER DEBUG ==========")
        print(f"UP count: {counter.up_count}")
        print(f"DOWN count: {counter.down_count}")
        print(f"CLASS COUNTS: {counter.class_counts}")
        print("========================================\n")
        summary = metrics.compute_summary(
            up_count=counter.up_count,
            down_count=counter.down_count,
            class_counts=counter.class_counts,
        )
        summary["output_video"] = str(output_video_path) if output_video_path else None
        return summary
