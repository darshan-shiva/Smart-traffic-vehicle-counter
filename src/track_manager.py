"""Track Management and Trajectory Processing Module.

Manages vehicle track lifecycles, age progression, minimum track age filtering,
rolling center-point trajectories, position history (previous/current),
moving-average center smoothing, and stale track eviction.

Author: Darshan S.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class TrackedVehicle:
    """Represents the complete state and trajectory history of a tracked vehicle."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]                                 # [x1, y1, x2, y2]
    center: Tuple[int, int]                           # Raw bounding box center
    bottom_point: Tuple[int, int]                     # Raw bottom ground-contact point
    current_center: Tuple[int, int]                   # Smoothed current center
    previous_center: Optional[Tuple[int, int]]        # Smoothed previous center (or None on first frame)
    current_bottom: Tuple[int, int]                   # Smoothed current bottom point
    previous_bottom: Optional[Tuple[int, int]]        # Smoothed previous bottom point (or None on first frame)
    age: int = 1                                      # Total frames since registration
    hits: int = 1                                     # Total frames detection matched
    time_since_update: int = 0                        # Consecutive frames since last detection
    is_confirmed: bool = False                        # Whether track meets min_age threshold
    history: List[Tuple[int, int]] = field(default_factory=list)  # Historical trajectory of center points

    def to_dict(self) -> Dict[str, Any]:
        """Convert track state to dictionary format for downstream processing."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "bottom_point": self.bottom_point,
            "current_center": self.current_center,
            "previous_center": self.previous_center,
            "current_bottom": self.current_bottom,
            "previous_bottom": self.previous_bottom,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "is_confirmed": self.is_confirmed,
            "trajectory": list(self.history),
            "history": list(self.history),
        }


class TrackManager:
    """Manages vehicle track lifecycles, trajectory histories, smoothing, and stale pruning."""

    # High-contrast color palette for track visualization
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
        min_age: int = 3,
        max_staleness: int = 15,
        history_len: int = 30,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize the TrackManager.

        Args:
            min_age: Minimum frames track must exist before being considered confirmed.
            max_staleness: Max consecutive frames without detection before track is pruned.
            history_len: Rolling history deque maximum capacity.
            smoothing_window: Number of recent points used for moving average smoothing.
        """
        self.min_age = max(1, min_age)
        self.max_staleness = max(1, max_staleness)
        self.history_len = max(2, history_len)
        self.smoothing_window = max(1, smoothing_window)

        # Internal track stores: track_id -> TrackedVehicle
        self._tracks: Dict[int, TrackedVehicle] = {}
        # Rolling deques for raw centers: track_id -> deque of (cx, cy)
        self._center_deques: Dict[int, Deque[Tuple[int, int]]] = {}
        # Rolling deques for raw bottom points: track_id -> deque of (bcx, bcy)
        self._bottom_deques: Dict[int, Deque[Tuple[int, int]]] = {}

        # Diagnostic counters
        self.total_tracks_created: int = 0
        self.total_stale_pruned: int = 0

    @property
    def active_tracks(self) -> Dict[int, TrackedVehicle]:
        """Dictionary of currently tracked vehicles."""
        return self._tracks

    def _compute_moving_average(
        self,
        points_deque: Deque[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Compute moving average over the most recent points in the deque.

        Args:
            points_deque: Deque containing coordinate tuples.

        Returns:
            Smoothed (x, y) integer coordinate tuple.
        """
        points = list(points_deque)
        recent = points[-self.smoothing_window:]
        avg_x = int(round(sum(p[0] for p in recent) / len(recent)))
        avg_y = int(round(sum(p[1] for p in recent) / len(recent)))
        return (avg_x, avg_y)

    def update(
        self,
        raw_tracks: List[Dict[str, Any]],
        filter_unconfirmed: bool = True,
    ) -> List[TrackedVehicle]:
        """Update track states with new observations from tracker.

        Args:
            raw_tracks: List of detection/track dicts from VehicleTracker.update().
            filter_unconfirmed: If True, returns only confirmed tracks (age >= min_age).

        Returns:
            List of TrackedVehicle objects active in the current frame.
        """
        current_frame_track_ids = set()

        # 1. Update existing tracks and register new ones
        for raw in raw_tracks:
            tid = int(raw["track_id"])
            current_frame_track_ids.add(tid)

            bbox = [float(c) for c in raw["bbox"]]
            conf = float(raw["confidence"])
            cls_id = int(raw.get("class_id", -1))
            cls_name = str(raw.get("class_name", "vehicle"))

            # Calculate raw centers if not present
            if "center" in raw:
                raw_cx, raw_cy = raw["center"]
            else:
                raw_cx = int(round((bbox[0] + bbox[2]) / 2.0))
                raw_cy = int(round((bbox[1] + bbox[3]) / 2.0))

            if "bottom_center" in raw:
                raw_bcx, raw_bcy = raw["bottom_center"]
            elif "bottom_point" in raw:
                raw_bcx, raw_bcy = raw["bottom_point"]
            else:
                raw_bcx = raw_cx
                raw_bcy = int(round(bbox[3]))

            raw_center = (raw_cx, raw_cy)
            raw_bottom = (raw_bcx, raw_bcy)

            if tid in self._tracks:
                # Existing track update
                track = self._tracks[tid]
                track.age += 1
                track.hits += 1
                track.time_since_update = 0
                track.bbox = bbox
                track.confidence = conf
                track.class_id = cls_id
                track.class_name = cls_name
                track.center = raw_center
                track.bottom_point = raw_bottom

                # Manage previous positions before updating current
                track.previous_center = track.current_center
                track.previous_bottom = track.current_bottom

                # Append to rolling deques
                self._center_deques[tid].append(raw_center)
                self._bottom_deques[tid].append(raw_bottom)

                # Compute new smoothed positions
                track.current_center = self._compute_moving_average(self._center_deques[tid])
                track.current_bottom = self._compute_moving_average(self._bottom_deques[tid])

                # Update trajectory history with smoothed coordinates
                track.history.append(track.current_center)
                if len(track.history) > self.history_len:
                    track.history.pop(0)

                # Confirmation threshold check
                if track.age >= self.min_age:
                    track.is_confirmed = True

            else:
                # New track initialization
                self.total_tracks_created += 1
                center_deq = deque([raw_center], maxlen=self.history_len)
                bottom_deq = deque([raw_bottom], maxlen=self.history_len)
                self._center_deques[tid] = center_deq
                self._bottom_deques[tid] = bottom_deq

                is_conf = (1 >= self.min_age)

                new_track = TrackedVehicle(
                    track_id=tid,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox,
                    center=raw_center,
                    bottom_point=raw_bottom,
                    current_center=raw_center,
                    previous_center=None,  # No previous position on birth frame
                    current_bottom=raw_bottom,
                    previous_bottom=None,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    is_confirmed=is_conf,
                    history=[raw_center],
                )
                self._tracks[tid] = new_track

        # 2. Increment staleness for tracks missing in this frame
        stale_ids = []
        for tid, track in self._tracks.items():
            if tid not in current_frame_track_ids:
                track.time_since_update += 1
                track.age += 1
                if track.time_since_update > self.max_staleness:
                    stale_ids.append(tid)

        # 3. Evict stale tracks
        for tid in stale_ids:
            del self._tracks[tid]
            self._center_deques.pop(tid, None)
            self._bottom_deques.pop(tid, None)
            self.total_stale_pruned += 1

        # 4. Filter output for the caller
        active_in_frame: List[TrackedVehicle] = [
            self._tracks[tid] for tid in current_frame_track_ids if tid in self._tracks
        ]

        if filter_unconfirmed:
            return [t for t in active_in_frame if t.is_confirmed]
        return active_in_frame

    def get_track(self, track_id: int) -> Optional[TrackedVehicle]:
        """Retrieve a specific track by its ID."""
        return self._tracks.get(track_id)

    def get_confirmed_tracks(self) -> List[TrackedVehicle]:
        """Return all currently active and confirmed tracks."""
        return [t for t in self._tracks.values() if t.is_confirmed and t.time_since_update == 0]

    def reset(self) -> None:
        """Reset all tracking history and state."""
        self._tracks.clear()
        self._center_deques.clear()
        self._bottom_deques.clear()
        self.total_tracks_created = 0
        self.total_stale_pruned = 0

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent, distinct BGR color for a given track ID."""
        return self.TRACK_PALETTE[track_id % len(self.TRACK_PALETTE)]

    def draw_managed_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackedVehicle],
        draw_boxes: bool = True,
        draw_trails: bool = True,
        draw_centers: bool = True,
        draw_vectors: bool = True,
    ) -> np.ndarray:
        """Render managed vehicle tracks with IDs, age, smoothed centers, and motion vectors.

        Args:
            frame: Input image array (BGR format).
            tracks: List of TrackedVehicle objects.
            draw_boxes: Whether to draw bounding boxes and labels.
            draw_trails: Whether to draw trajectory history polylines.
            draw_centers: Whether to draw smoothed center dots.
            draw_vectors: Whether to draw displacement vectors from previous to current center.

        Returns:
            Annotated frame.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid numpy ndarray.")

        annotated = frame.copy()

        for track in tracks:
            tid = track.track_id
            cls_name = track.class_name
            conf = track.confidence
            color = self.get_track_color(tid)

            # 1. Bounding Box & Label
            if draw_boxes:
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"ID #{tid} {cls_name} (age:{track.age})"
                font_scale = 0.50
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

            # 2. Historical Trajectory Trail
            if draw_trails and len(track.history) > 1:
                pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

            # 3. Center Points (Smoothed = Red circle, Raw = small dark dot)
            if draw_centers:
                scx, scy = track.current_center
                cv2.circle(annotated, (scx, scy), 5, (0, 0, 255), -1)  # Smoothed center
                rcx, rcy = track.center
                cv2.circle(annotated, (rcx, rcy), 2, (0, 0, 0), -1)    # Raw center

            # 4. Previous to Current Motion Vector (Yellow arrow)
            if draw_vectors and track.previous_center is not None:
                px, py = track.previous_center
                cx, cy = track.current_center
                cv2.arrowedLine(
                    annotated,
                    (px, py),
                    (cx, cy),
                    (0, 255, 255),
                    2,
                    tipLength=0.3,
                )

        return annotated
