"""Multi-Object Vehicle Tracker Module.

Provides abstract tracking interface and implementations for ByteTrack and BoT-SORT
using Ultralytics, with persistent track IDs, trajectory smoothing, and visual annotations.

Author: Darshan S.
"""

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.detector import VehicleDetector


class BaseTracker(ABC):
    """Abstract base class defining the vehicle tracking interface."""

    def __init__(
        self,
        tracker_type: str,
        config_path: Optional[Union[str, Path]] = None,
        min_hits: int = 1,
        trajectory_history: int = 30,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize the base tracker.

        Args:
            tracker_type: Identifier of the tracking algorithm ('bytetrack' or 'botsort').
            config_path: Path to tracker YAML config file.
            min_hits: Number of consecutive detections required before track is confirmed.
            trajectory_history: Maximum number of trajectory points to retain.
            smoothing_window: Window size for moving-average trajectory smoothing.
        """
        self.tracker_type = tracker_type.lower()
        self.config_path = self._resolve_config(config_path)
        self.min_hits = max(1, min_hits)
        self.trajectory_history = max(2, trajectory_history)
        self.smoothing_window = max(1, smoothing_window)

        # State tracking: track_id -> deque of (cx, cy) points
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = {}
        # Hit counter: track_id -> consecutive frames detected
        self.track_hits: Dict[int, int] = {}

    def _resolve_config(self, config_path: Optional[Union[str, Path]]) -> str:
        """Resolve the tracker configuration file path.

        Args:
            config_path: Custom config path if provided.

        Returns:
            Resolved configuration path as a string.
        """
        if config_path is not None:
            p = Path(config_path)
            if p.is_file():
                return str(p.resolve())
            return str(config_path)

        local_cfg = Path("configs") / f"{self.tracker_type}.yaml"
        if local_cfg.is_file():
            return str(local_cfg.resolve())

        # Fallback to Ultralytics built-in configuration
        return f"{self.tracker_type}.yaml"

    @abstractmethod
    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a video frame and return active vehicle tracks.

        Args:
            frame: Input video frame (BGR format).

        Returns:
            List of track dictionaries containing:
                - 'track_id': int
                - 'class_id': int
                - 'class_name': str
                - 'confidence': float
                - 'bbox': [x1, y1, x2, y2]
                - 'center': (cx, cy)
                - 'smoothed_center': (scx, scy)
                - 'bottom_center': (bcx, bcy)
                - 'trajectory': List[Tuple[int, int]]
        """
        pass

    def reset(self) -> None:
        """Reset internal tracking state and history."""
        self.trajectories.clear()
        self.track_hits.clear()

    def _smooth_center(self, track_id: int, current_center: Tuple[int, int]) -> Tuple[int, int]:
        """Compute moving-average smoothed center coordinate for a track.

        Args:
            track_id: Unique track identifier.
            current_center: Current frame (cx, cy) center point.

        Returns:
            Smoothed (x, y) coordinate tuple.
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.trajectory_history)

        self.trajectories[track_id].append(current_center)
        points = list(self.trajectories[track_id])
        recent = points[-self.smoothing_window:]

        avg_x = int(round(sum(p[0] for p in recent) / len(recent)))
        avg_y = int(round(sum(p[1] for p in recent) / len(recent)))
        return (avg_x, avg_y)


class UltralyticsTracker(BaseTracker):
    """General tracker backend utilizing Ultralytics YOLO tracking engine."""

    def __init__(
        self,
        tracker_type: str,
        detector: Optional[VehicleDetector] = None,
        config_path: Optional[Union[str, Path]] = None,
        min_hits: int = 1,
        trajectory_history: int = 30,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize the Ultralytics-based tracker.

        Args:
            tracker_type: Algorithm name ('bytetrack' or 'botsort').
            detector: Existing VehicleDetector instance. If None, one is created.
            config_path: Optional custom config path.
            min_hits: Minimum detection count before track confirmation.
            trajectory_history: Number of trailing points preserved.
            smoothing_window: Moving average smoothing window size.
        """
        super().__init__(
            tracker_type=tracker_type,
            config_path=config_path,
            min_hits=min_hits,
            trajectory_history=trajectory_history,
            smoothing_window=smoothing_window,
        )

        self.detector = detector if detector is not None else VehicleDetector()
        self.model = self.detector.model
        self.target_classes = self.detector.TARGET_VEHICLE_CLASSES
        self.device = self.detector.device
        self.confidence_threshold = self.detector.confidence_threshold
        self.iou_threshold = self.detector.iou_threshold

    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracker with a new frame.

        Args:
            frame: Input image array (BGR format).

        Returns:
            List of confirmed active vehicle tracks.
        """
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim not in (2, 3)
        ):
            return []

        target_ids = list(self.target_classes.keys())
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=target_ids,
            device=self.device,
            verbose=False,
        )

        tracks: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return tracks

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return tracks

        # Extract coordinates, IDs, confidences, and classes
        track_ids = boxes.id.cpu().numpy().astype(int)
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        current_frame_ids = set()

        for i in range(len(track_ids)):
            cls_id = int(class_ids[i])
            if cls_id not in self.target_classes:
                continue

            tid = int(track_ids[i])
            current_frame_ids.add(tid)
            self.track_hits[tid] = self.track_hits.get(tid, 0) + 1

            # Ignore transient tracks below min_hits threshold
            if self.track_hits[tid] < self.min_hits:
                continue

            x1, y1, x2, y2 = xyxy_coords[i]
            conf = float(confidences[i])
            cls_name = self.target_classes[cls_id]

            raw_cx = int(round((x1 + x2) / 2.0))
            raw_cy = int(round((y1 + y2) / 2.0))
            smoothed_center = self._smooth_center(tid, (raw_cx, raw_cy))

            bcx = raw_cx
            bcy = int(round(y2))

            tracks.append({
                "track_id": tid,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": (raw_cx, raw_cy),
                "smoothed_center": smoothed_center,
                "bottom_center": (bcx, bcy),
                "trajectory": list(self.trajectories.get(tid, [])),
            })

        return tracks

    def reset(self) -> None:
        """Reset internal tracking state and model predictor trackers."""
        super().reset()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []


class ByteTrackTracker(UltralyticsTracker):
    """Specialized ByteTrack multi-object vehicle tracker."""

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        config_path: Optional[Union[str, Path]] = None,
        min_hits: int = 1,
        trajectory_history: int = 30,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize ByteTrack tracker."""
        super().__init__(
            tracker_type="bytetrack",
            detector=detector,
            config_path=config_path,
            min_hits=min_hits,
            trajectory_history=trajectory_history,
            smoothing_window=smoothing_window,
        )


class BotSortTracker(UltralyticsTracker):
    """Specialized BoT-SORT multi-object vehicle tracker."""

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        config_path: Optional[Union[str, Path]] = None,
        min_hits: int = 1,
        trajectory_history: int = 30,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize BoT-SORT tracker."""
        super().__init__(
            tracker_type="botsort",
            detector=detector,
            config_path=config_path,
            min_hits=min_hits,
            trajectory_history=trajectory_history,
            smoothing_window=smoothing_window,
        )


class VehicleTracker:
    """Unified multi-object vehicle tracking manager and visualizer."""

    SUPPORTED_TRACKERS = {
        "bytetrack": ByteTrackTracker,
        "botsort": BotSortTracker,
    }

    # Distinct high-visibility colors for tracks
    TRACK_PALETTE: List[Tuple[int, int, int]] = [
        (255, 105, 65),   # Coral Orange
        (0, 215, 255),   # Golden Yellow
        (50, 205, 50),   # Lime Green
        (238, 130, 238), # Violet
        (30, 144, 255),  # Dodger Blue
        (255, 20, 147),  # Deep Pink
        (0, 255, 255),   # Cyan
        (255, 165, 0),   # Orange
        (138, 43, 226),  # Blue Violet
        (0, 250, 154),   # Medium Spring Green
    ]

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        detector: Optional[VehicleDetector] = None,
        config_path: Optional[Union[str, Path]] = None,
        min_hits: int = 1,
        trajectory_history: int = 30,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize the unified vehicle tracker.

        Args:
            tracker_type: Tracking algorithm ('bytetrack' or 'botsort').
            detector: Existing VehicleDetector instance.
            config_path: Optional custom config file path.
            min_hits: Consecutive frames before confirming a track.
            trajectory_history: Rolling trajectory history length.
            smoothing_window: Moving average smoothing window size.

        Raises:
            ValueError: If tracker_type is not supported.
        """
        self.tracker_name = tracker_type.lower()
        if self.tracker_name not in self.SUPPORTED_TRACKERS:
            raise ValueError(
                f"Unsupported tracker '{tracker_type}'. "
                f"Choose from: {list(self.SUPPORTED_TRACKERS.keys())}"
            )

        tracker_cls = self.SUPPORTED_TRACKERS[self.tracker_name]
        self._tracker: BaseTracker = tracker_cls(
            detector=detector,
            config_path=config_path,
            min_hits=min_hits,
            trajectory_history=trajectory_history,
            smoothing_window=smoothing_window,
        )

    @property
    def tracker(self) -> BaseTracker:
        """Access the underlying tracker instance."""
        return self._tracker

    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracker with frame and return active tracks."""
        return self._tracker.update(frame)

    def reset(self) -> None:
        """Reset tracking state."""
        self._tracker.reset()

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a consistent, distinct color for a given track ID."""
        return self.TRACK_PALETTE[track_id % len(self.TRACK_PALETTE)]

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        draw_trails: bool = True,
        draw_centers: bool = True,
        draw_bottom_center: bool = True,
    ) -> np.ndarray:
        """Render track bounding boxes, IDs, labels, and motion trails onto frame.

        Args:
            frame: Input image array (BGR format).
            tracks: List of active track dictionaries from update().
            draw_trails: Whether to draw historical trajectory trails.
            draw_centers: Whether to draw smoothed center dots.
            draw_bottom_center: Whether to draw ground-contact cyan dots.

        Returns:
            Annotated frame.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid numpy ndarray.")

        annotated = frame.copy()

        for trk in tracks:
            tid = trk["track_id"]
            cls_name = trk["class_name"]
            conf = trk["confidence"]
            x1, y1, x2, y2 = map(int, trk["bbox"])
            color = self.get_track_color(tid)

            # 1. Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 2. Track label banner with ID and class
            label = f"ID #{tid} {cls_name} {conf:.2f}"
            font_scale = 0.52
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            text_y1 = max(0, y1 - text_h - 6)
            text_y2 = y1
            cv2.rectangle(annotated, (x1, text_y1), (x1 + text_w + 6, text_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

            # 3. Trajectory trail
            trajectory = trk.get("trajectory", [])
            if draw_trails and len(trajectory) > 1:
                pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

            # 4. Smoothed center point (red circle)
            if draw_centers:
                scx, scy = trk["smoothed_center"]
                cv2.circle(annotated, (scx, scy), 4, (0, 0, 255), -1)

            # 5. Bottom ground-contact point (cyan circle)
            if draw_bottom_center:
                bcx, bcy = trk["bottom_center"]
                cv2.circle(annotated, (bcx, bcy), 4, (255, 255, 0), -1)

        return annotated
