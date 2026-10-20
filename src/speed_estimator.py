from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class SpeedRecord:
    track_id: int
    speed_kmh: float
    elapsed_seconds: float


class SpeedEstimator:
    """
    Estimates vehicle speed using two virtual lines.

    Speed = real-world distance / time taken between two lines.
    """

    def __init__(
        self,
        line1_y: int,
        line2_y: int,
        distance_meters: float = 10.0,
        fps: float = 30.0,
        max_reasonable_speed_kmh: float = 150.0,
    ):
        self.line1_y = line1_y
        self.line2_y = line2_y
        self.distance_meters = distance_meters
        self.fps = fps
        self.max_reasonable_speed_kmh = max_reasonable_speed_kmh

        # Track -> frame when first line was crossed
        self.line1_crossings: Dict[int, int] = {}

        # Track -> frame when second line was crossed
        self.line2_crossings: Dict[int, int] = {}

        # Final speed records
        self.speed_records: Dict[int, SpeedRecord] = {}

    def _crossed_line(
        self,
        previous_y: float,
        current_y: float,
        line_y: int,
    ) -> bool:

        return (
            previous_y < line_y <= current_y
            or
            previous_y > line_y >= current_y
        )

    def update_track(
        self,
        track_id: int,
        previous_y: float,
        current_y: float,
        frame_number: int,
    ) -> Optional[SpeedRecord]:

        # First line
        if (
            track_id not in self.line1_crossings
            and self._crossed_line(
                previous_y,
                current_y,
                self.line1_y,
            )
        ):
            self.line1_crossings[track_id] = frame_number

            return None

        # Second line
        if (
            track_id in self.line1_crossings
            and track_id not in self.line2_crossings
            and self._crossed_line(
                previous_y,
                current_y,
                self.line2_y,
            )
        ):

            self.line2_crossings[track_id] = frame_number

            first_frame = self.line1_crossings[track_id]

            frame_difference = frame_number - first_frame

            if frame_difference <= 0 or self.fps <= 0:
                return None

            elapsed_seconds = frame_difference / self.fps

            speed_mps = self.distance_meters / elapsed_seconds
            speed_kmh = speed_mps * 3.6

            # Reject unrealistic values
            if speed_kmh <= 0:
                return None

            if speed_kmh > self.max_reasonable_speed_kmh:
                return None

            record = SpeedRecord(
                track_id=track_id,
                speed_kmh=round(speed_kmh, 2),
                elapsed_seconds=round(elapsed_seconds, 3),
            )

            self.speed_records[track_id] = record

            return record

        return None

    def get_speeds(self) -> List[float]:

        return [
            record.speed_kmh
            for record in self.speed_records.values()
        ]

    def get_average_speed(self) -> float:

        speeds = self.get_speeds()

        if not speeds:
            return 0.0

        return sum(speeds) / len(speeds)

    def get_max_speed(self) -> float:

        speeds = self.get_speeds()

        if not speeds:
            return 0.0

        return max(speeds)

    def get_summary(self) -> dict:
        speeds = self.get_speeds()

        speed_records = [
            {
                "track_id": record.track_id,
                "speed_kmh": record.speed_kmh,
                "elapsed_seconds": record.elapsed_seconds,
            }
            for record in self.speed_records.values()
        ]

        return {
            "Speed Estimates": len(speeds),
            "Average Speed (km/h)": round(
                self.get_average_speed(),
                2,
            ),
            "Maximum Speed (km/h)": round(
                self.get_max_speed(),
                2,
            ),
            "Speed Distance (m)": self.distance_meters,
            "Speed Records": speed_records,
        }