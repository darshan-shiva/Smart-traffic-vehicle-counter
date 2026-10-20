"""Vehicle Detector Module.

Uses Ultralytics YOLO11 to detect vehicle classes (car, motorcycle, bus, truck)
in traffic surveillance video frames.

Author: Darshan S.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    torch = None  # type: ignore
    YOLO = None   # type: ignore


class VehicleDetector:
    """YOLO11 wrapper specialized for traffic vehicle detection.
    
    Target classes are constrained to common vehicular traffic:
        - 2: car
        - 3: motorcycle
        - 5: bus
        - 7: truck
    """

    TARGET_VEHICLE_CLASSES: Dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # High-contrast color palette (BGR) for visual clarity
    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "car": (0, 215, 255),         # Vibrant Gold/Amber
        "motorcycle": (255, 128, 0),  # Azure Blue
        "bus": (76, 177, 34),         # Emerald Green
        "truck": (204, 50, 153),      # Violet / Purple
    }

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the YOLO11 vehicle detector.

        Args:
            model_path: Model name (e.g. 'yolo11n.pt', 'yolo11s.pt') or Path to local weights.
            confidence_threshold: Minimum detection confidence threshold (0.0 to 1.0).
            iou_threshold: Non-Maximum Suppression (NMS) IoU threshold (0.0 to 1.0).
            device: Computation device ('cpu', 'cuda', '0', etc.). If None, auto-selects.

        Raises:
            ImportError: If ultralytics or torch are not installed.
            ValueError: If confidence_threshold or iou_threshold are out of bounds.
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required but not installed. "
                "Install it using: pip install ultralytics"
            )

        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}")
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}")

        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

        # Select compute device
        if device is None:
            self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Load YOLO11 model safely with informative error messages.

        Returns:
            Loaded YOLO model instance.
        """
        # Check if model exists directly or inside models/ directory
        candidate_path = self.model_path
        if not candidate_path.exists():
            models_dir_candidate = Path("models") / candidate_path.name
            if models_dir_candidate.exists():
                candidate_path = models_dir_candidate

        model_identifier = str(candidate_path)
        try:
            model = YOLO(model_identifier)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_identifier}'. "
                f"Ensure the model file exists or you have internet access to download weights. "
                f"Error: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection on an image or video frame.

        Args:
            frame: Input image array in BGR format (height, width, 3).

        Returns:
            List of detected vehicle dictionaries containing:
                - 'bbox': [x1, y1, x2, y2] (float pixel coordinates)
                - 'confidence': float score
                - 'class_id': int COCO class id
                - 'class_name': str vehicle label
                - 'center': (cx, cy) bounding box center
                - 'bottom_center': (bcx, bcy) vehicle ground contact proxy
        """
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim not in (2, 3)
        ):
            return []

        # Run YOLO inference filtering for vehicular classes only
        target_ids = list(self.TARGET_VEHICLE_CLASSES.keys())
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=target_ids,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        first_result = results[0]
        boxes = first_result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract tensor data to CPU numpy arrays
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy_coords)):
            cls_id = int(class_ids[i])
            if cls_id not in self.TARGET_VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = xyxy_coords[i]
            conf = float(confidences[i])
            class_name = self.TARGET_VEHICLE_CLASSES[cls_id]

            # Calculate center points
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            bcx = cx
            bcy = int(round(y2))

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": class_name,
                "center": (cx, cy),
                "bottom_center": (bcx, bcy),
            })

        return detections

    def detect_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load an image file from disk and detect vehicles.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (loaded BGR image ndarray, detections list).

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image file cannot be decoded.
        """
        import cv2

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to decode image from: {path}")

        return image, self.detect(image)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_center: bool = True,
        draw_bottom_center: bool = True,
    ) -> np.ndarray:
        """Render detection bounding boxes, labels, and center points onto a frame.

        Args:
            frame: Input image array (BGR format).
            detections: List of detection dictionaries returned by detect().
            draw_center: If True, draws a red dot at the vehicle bounding box center.
            draw_bottom_center: If True, draws a cyan dot at the ground contact point.

        Returns:
            Annotated frame (copy of original).
        """
        import cv2

        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid numpy ndarray.")

        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class_name"]
            conf = det["confidence"]
            color = self.CLASS_COLORS.get(class_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text & background
            label = f"{class_name} {conf:.2f}"
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw label banner
            text_y1 = max(0, y1 - text_h - 6)
            text_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, text_y1),
                (x1 + text_w + 6, text_y2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )

            # Bounding box center point (Red dot)
            if draw_center:
                cx, cy = det["center"]
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            # Ground contact point (Cyan dot)
            if draw_bottom_center:
                bcx, bcy = det["bottom_center"]
                cv2.circle(annotated, (bcx, bcy), 5, (255, 255, 0), -1)

        return annotated
