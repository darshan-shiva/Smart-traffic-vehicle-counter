"""Unit and Integration Tests for VehicleDetector Module.

Author: Darshan S.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector, ULTRALYTICS_AVAILABLE


def test_detector_classes() -> None:
    """Verify target vehicle classes are properly defined."""
    assert VehicleDetector.TARGET_VEHICLE_CLASSES[2] == "car"
    assert VehicleDetector.TARGET_VEHICLE_CLASSES[3] == "motorcycle"
    assert VehicleDetector.TARGET_VEHICLE_CLASSES[5] == "bus"
    assert VehicleDetector.TARGET_VEHICLE_CLASSES[7] == "truck"
    print("✓ Target vehicle classes verified successfully.")


def test_detector_threshold_validation() -> None:
    """Verify confidence and IoU threshold boundaries."""
    if not ULTRALYTICS_AVAILABLE:
        print("Ultralytics not installed, skipping threshold test.")
        return

    try:
        VehicleDetector(confidence_threshold=-0.1)
        assert False, "Should have raised ValueError for negative confidence"
    except ValueError:
        pass

    try:
        VehicleDetector(iou_threshold=1.5)
        assert False, "Should have raised ValueError for IoU > 1.0"
    except ValueError:
        pass

    print("✓ Threshold parameter validation verified successfully.")


def test_detector_device_detection() -> None:
    """Verify device selection logic (CPU or CUDA)."""
    detector = VehicleDetector(model_path="models/yolo11n.pt")
    assert detector.device in ("cpu", "cuda") or "cuda" in detector.device
    print(f"✓ Device detection verified: {detector.device}")


def test_detector_sample_inference() -> None:
    """Verify vehicle detection, class filtering, and coordinate extraction on sample image."""
    sample_path = PROJECT_ROOT / "assets" / "traffic_sample.jpg"
    assert sample_path.exists(), f"Sample image missing at {sample_path}"

    detector = VehicleDetector(model_path="models/yolo11n.pt", confidence_threshold=0.35)
    image, detections = detector.detect_image(sample_path)

    assert len(detections) > 0, "Expected at least 1 detection on sample image"
    valid_classes = {"car", "motorcycle", "bus", "truck"}

    for det in detections:
        assert det["class_name"] in valid_classes, f"Invalid class detected: {det['class_name']}"
        assert 0.0 <= det["confidence"] <= 1.0
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 < x2 <= image.shape[1]
        assert 0 <= y1 < y2 <= image.shape[0]

        expected_cx = int(round((x1 + x2) / 2.0))
        expected_cy = int(round((y1 + y2) / 2.0))
        assert det["center"] == (expected_cx, expected_cy)

        expected_bcx = expected_cx
        expected_bcy = int(round(y2))
        assert det["bottom_center"] == (expected_bcx, expected_bcy)

    print(f"✓ Inference verified on sample image: {len(detections)} vehicles detected.")

    # Annotation test
    annotated = detector.draw_detections(image, detections)
    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype

    # Output save test
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "phase1_detector_test.jpg"
    assert cv2.imwrite(str(out_file), annotated)
    assert out_file.exists() and out_file.stat().st_size > 0
    print("✓ Output annotation image generated and verified.")


def test_detector_edge_cases() -> None:
    """Verify edge cases: missing file, blank frame, malformed inputs."""
    detector = VehicleDetector(model_path="models/yolo11n.pt")

    # Missing file
    try:
        detector.detect_image("non_existent_file.jpg")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass

    # Blank frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) == []

    # Malformed inputs
    assert detector.detect(None) == []  # type: ignore
    assert detector.detect(np.zeros((0,), dtype=np.uint8)) == []
    assert detector.detect(np.zeros((10,), dtype=np.uint8)) == []

    # CPU explicit run
    cpu_detector = VehicleDetector(model_path="models/yolo11n.pt", device="cpu")
    assert cpu_detector.device == "cpu"

    print("✓ All edge cases verified successfully.")


if __name__ == "__main__":
    print("--- Running Complete VehicleDetector Test Suite ---")
    test_detector_classes()
    test_detector_threshold_validation()
    test_detector_device_detection()
    test_detector_sample_inference()
    test_detector_edge_cases()
    print("--- All Tests Passed Successfully ---")
