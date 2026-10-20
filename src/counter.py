"""Vehicle Counting and Line-Crossing Logic Module.

Tracks center-point crossings across a configurable horizontal line,
categorizes motion direction (UP/DOWN), prevents duplicate counts,
and maintains a neutral margin zone to eliminate jitter.

Author: Darshan S.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

from src.track_manager import TrackedVehicle


class VehicleCounter:
    """Handles line crossing detection and bidirectional counting."""

    def __init__(
        self,
        line_position_y: Optional[int] = None,
        line_ratio: float = 0.55,
        frame_height: Optional[int] = None,
        margin: int = 10,
        line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> None:
        """Initialize the vehicle counter.

        Args:
            line_position_y: Explicit Y-coordinate of the counting line.
            line_ratio: Normalized vertical position (0.0 to 1.0) if line_position_y is None.
            frame_height: Video frame height used to resolve line_ratio.
            margin: Neutral zone half-width in pixels (±margin around line_position_y).
            line_coords: Optional explicit ((x1, y1), (x2, y2)) line endpoints.
        """
        self.line_ratio = float(line_ratio)
        self.margin = max(1, int(margin))
        self.line_coords = line_coords

        if line_position_y is not None:
            self._line_y = int(line_position_y)
        elif frame_height is not None:
            self._line_y = int(round(frame_height * self.line_ratio))
        else:
            self._line_y = None

        # Directional and class count tallies
        self.up_count: int = 0
        self.down_count: int = 0
        self.class_counts: Dict[str, Dict[str, int]] = {
            "car": {"up": 0, "down": 0},
            "bus": {"up": 0, "down": 0},
            "truck": {"up": 0, "down": 0},
            "motorcycle": {"up": 0, "down": 0},
        }

        # Duplicate prevention & state tracking
        self.counted_ids: Set[int] = set()
        # track_id -> "ABOVE" | "BELOW" (origin side before crossing)
        self.initial_sides: Dict[int, str] = {}
        # Recent crossing events: track_id -> (direction, frame_timestamp_or_idx)
        self.recent_crossings: Dict[int, Dict[str, Any]] = {}

    @property
    def line_position_y(self) -> int:
        """Current Y-coordinate of the horizontal counting line."""
        if self._line_y is None:
            raise ValueError("Line position Y has not been initialized. Provide frame_height or line_position_y.")
        return self._line_y

    @line_position_y.setter
    def line_position_y(self, y: int) -> None:
        """Set explicit Y-coordinate for the counting line."""
        self._line_y = int(y)

    @property
    def total_count(self) -> int:
        """Total number of vehicles counted (always equals up_count + down_count)."""
        return self.up_count + self.down_count

    def resolve_line_y(self, frame_height: int) -> int:
        """Resolve or confirm line Y position using the given frame height.

        Args:
            frame_height: Height of the current video frame.

        Returns:
            Resolved integer Y coordinate.
        """
        if self._line_y is None:
            self._line_y = int(round(frame_height * self.line_ratio))
        return self._line_y

    def is_above(self, y: int) -> bool:
        """Check if a Y-coordinate is strictly above the neutral margin zone."""
        return y < (self.line_position_y - self.margin)

    def is_below(self, y: int) -> bool:
        """Check if a Y-coordinate is strictly below the neutral margin zone."""
        return y > (self.line_position_y + self.margin)

    def is_neutral(self, y: int) -> bool:
        """Check if a Y-coordinate falls within the neutral margin zone."""
        return abs(y - self.line_position_y) <= self.margin

    def check_crossing(
        self,
        track_id: int,
        class_name: str,
        current_center: Tuple[int, int],
        previous_center: Optional[Tuple[int, int]],
    ) -> Optional[str]:
        """Evaluate if a tracked vehicle has crossed the counting line.

        Rules:
            1. If track_id has already been counted, immediately return None (single-count guarantee).
            2. Requires previous_center to establish motion direction.
            3. Uses neutral margin zone to eliminate jitter crossings.
            4. In image coordinates, Y increases downwards:
               - Moving UP: Previous Y > Line Y and Current Y <= Line Y (or from BELOW to ABOVE)
               - Moving DOWN: Previous Y < Line Y and Current Y >= Line Y (or from ABOVE to BELOW)

        Args:
            track_id: Persistent track identifier.
            class_name: Vehicle class category.
            current_center: Current (x, y) center point.
            previous_center: Previous (x, y) center point.

        Returns:
            "UP", "DOWN", or None.
        """
        # Rule 1: Never count the same track ID more than once
        if track_id in self.counted_ids:
            return None

        if previous_center is None:
            # First frame observed: establish origin side if outside neutral zone
            cy = current_center[1]
            if self.is_above(cy):
                self.initial_sides[track_id] = "ABOVE"
            elif self.is_below(cy):
                self.initial_sides[track_id] = "BELOW"
            return None

        prev_y = previous_center[1]
        curr_y = current_center[1]
        line_y = self.line_position_y

        # Update initial side if not set yet (must be strictly outside neutral zone)
        if track_id not in self.initial_sides:
            if self.is_above(prev_y):
                self.initial_sides[track_id] = "ABOVE"
            elif self.is_below(prev_y):
                self.initial_sides[track_id] = "BELOW"

        origin = self.initial_sides.get(track_id)
        crossing_direction: Optional[str] = None

        # Check DOWN crossing: originated strictly ABOVE neutral zone (y < line_y - margin)
        # and has now crossed line_y
        if origin == "ABOVE" and prev_y < line_y and curr_y >= line_y:
            crossing_direction = "DOWN"

        # Check UP crossing: originated strictly BELOW neutral zone (y > line_y + margin)
        # and has now crossed line_y
        elif origin == "BELOW" and prev_y > line_y and curr_y <= line_y:
            crossing_direction = "UP"

        if crossing_direction is not None:
            # Register count
            self.counted_ids.add(track_id)

            if crossing_direction == "UP":
                self.up_count += 1
            else:
                self.down_count += 1

            # Update class tally
            normalized_cls = class_name.lower()
            if normalized_cls not in self.class_counts:
                self.class_counts[normalized_cls] = {"up": 0, "down": 0}
            self.class_counts[normalized_cls][crossing_direction.lower()] += 1
            print(
                f"[CLASS COUNT] {normalized_cls} -> "
                f"{self.class_counts[normalized_cls]}"
            )

            # Record event
            self.recent_crossings[track_id] = {
                "direction": crossing_direction,
                "class_name": class_name,
                "position": current_center,
            }

            return crossing_direction

        return None

    def update(
        self,
        tracks: Union[List[TrackedVehicle], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Update counter state with all active tracks in the current frame.

        Args:
            tracks: List of TrackedVehicle objects or track dictionaries.

        Returns:
            List of crossing event dictionaries that occurred in this frame.
        """
        crossing_events: List[Dict[str, Any]] = []

        for trk in tracks:
            if isinstance(trk, TrackedVehicle):
                tid = trk.track_id
                cls_name = trk.class_name
                curr_c = trk.current_center
                prev_c = trk.previous_center
            else:
                tid = int(trk["track_id"])
                cls_name = str(trk.get("class_name", "vehicle"))
                curr_c = trk["current_center"] if "current_center" in trk else trk["center"]
                prev_c = trk.get("previous_center")

            direction = self.check_crossing(
                track_id=tid,
                class_name=cls_name,
                current_center=curr_c,
                previous_center=prev_c,
            )

            if direction is not None:
                crossing_events.append({
                    "track_id": tid,
                    "class_name": cls_name,
                    "direction": direction,
                    "current_center": curr_c,
                    "previous_center": prev_c,
                })

        return crossing_events

    def reset(self) -> None:
        """Reset all counters, tracked IDs, and crossing state."""
        self.up_count = 0
        self.down_count = 0
        for cls_dict in self.class_counts.values():
            cls_dict["up"] = 0
            cls_dict["down"] = 0
        self.counted_ids.clear()
        self.initial_sides.clear()
        self.recent_crossings.clear()

    def draw_counter(
        self,
        frame: np.ndarray,
        tracks: Optional[List[TrackedVehicle]] = None,
    ) -> np.ndarray:
        """Render the counting line, neutral margin zone, and HUD count dashboard.

        Args:
            frame: Input video frame (BGR format).
            tracks: Optional list of TrackedVehicle objects to highlight recently counted.

        Returns:
            Annotated frame.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid numpy ndarray.")

        height, width = frame.shape[:2]
        line_y = self.resolve_line_y(height)
        annotated = frame.copy()

        # 1. Draw Neutral Margin Zone (Translucent grey band)
        y_top = max(0, line_y - self.margin)
        y_bottom = min(height, line_y + self.margin)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, y_top), (width, y_bottom), (70, 70, 70), -1)
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        # 2. Draw Counting Line (Bright Yellow/Amber with dashed guide margins)
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 215, 255), 2, cv2.LINE_AA)
        cv2.line(annotated, (0, y_top), (width, y_top), (150, 150, 150), 1, cv2.LINE_4)
        cv2.line(annotated, (0, y_bottom), (width, y_bottom), (150, 150, 150), 1, cv2.LINE_4)

        # Line label
        line_label = f"COUNTING LINE (Y={line_y}, Margin=+/-{self.margin}px)"
        cv2.putText(
            annotated,
            line_label,
            (15, line_y - self.margin - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        # 3. Draw Top HUD Dashboard
        # Solid dark background banner for readability
        hud_w = 420
        hud_h = 100
        cv2.rectangle(annotated, (width - hud_w - 15, 12), (width - 15, 12 + hud_h), (25, 25, 25), -1)
        cv2.rectangle(annotated, (width - hud_w - 15, 12), (width - 15, 12 + hud_h), (0, 215, 255), 1)

        # HUD Title
        cv2.putText(
            annotated,
            "TRAFFIC VEHICLE COUNTER",
            (width - hud_w, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # UP COUNT (Green)
        up_text = f"UP: {self.up_count}"
        cv2.putText(
            annotated,
            up_text,
            (width - hud_w, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 205, 50),
            2,
            cv2.LINE_AA,
        )

        # DOWN COUNT (Orange/Red)
        down_text = f"DOWN: {self.down_count}"
        cv2.putText(
            annotated,
            down_text,
            (width - hud_w + 140, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )

        # TOTAL COUNT (Cyan/White)
        total_text = f"TOTAL: {self.total_count}"
        cv2.putText(
            annotated,
            total_text,
            (width - hud_w + 290, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Breakdown summary
        breakdown = (
            f"Cars: {self.class_counts.get('car', {}).get('up', 0) + self.class_counts.get('car', {}).get('down', 0)} | "
            f"Trucks: {self.class_counts.get('truck', {}).get('up', 0) + self.class_counts.get('truck', {}).get('down', 0)} | "
            f"Buses: {self.class_counts.get('bus', {}).get('up', 0) + self.class_counts.get('bus', {}).get('down', 0)} | "
            f"Motos: {self.class_counts.get('motorcycle', {}).get('up', 0) + self.class_counts.get('motorcycle', {}).get('down', 0)}"
        )
        cv2.putText(
            annotated,
            breakdown,
            (width - hud_w, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # 4. Highlight counted vehicles
        if tracks is not None:
            for trk in tracks:
                if trk.track_id in self.counted_ids:
                    cx, cy = trk.current_center
                    cv2.circle(annotated, (cx, cy), 9, (0, 255, 0), 2)

        return annotated
