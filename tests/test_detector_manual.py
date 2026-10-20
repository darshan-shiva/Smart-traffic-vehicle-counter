"""Manual and End-to-End Verification Script for Phase 1: VehicleDetector.

Author: Darshan S.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector


def run_phase1_verification() -> bool:
    """Execute all Phase 1 verification tests with detailed logging."""
    results = {}
    print("=" * 60)
    print("STARTING PHASE 1 VERIFICATION FOR VEHICLEDETECTOR")
    print("Author: Darshan S.")
    print("=" * 60)

    # 1. Model Loading
    print("\n[Step 1] Loading YOLO11 Model...")
    try:
        model_path = PROJECT_ROOT / "models" / "yolo11n.pt"
        if not model_path.exists():
            model_path = "yolo11n.pt"
        detector = VehicleDetector(model_path=str(model_path), confidence_threshold=0.35)
        results["Model loading"] = True
        print(f"✓ Model successfully loaded from: {detector.model_path}")
    except Exception as exc:
        results["Model loading"] = False
        print(f"✗ Model loading failed: {exc}")
        return False

    # 2. Device Detection
    print("\n[Step 2] Checking Compute Device...")
    try:
        device_str = str(detector.device).upper()
        print(f"✓ Active Compute Device: {device_str}")
        results["Device detection"] = True
    except Exception as exc:
        results["Device detection"] = False
        print(f"✗ Device detection failed: {exc}")

    # 3. Load Test Image and Run Detection
    print("\n[Step 3] Loading Test Image & Running Detection...")
    test_image_path = PROJECT_ROOT / "assets" / "traffic_sample.jpg"
    if not test_image_path.exists():
        print(f"✗ Test image not found at: {test_image_path}")
        results["Vehicle filtering"] = False
        return False

    image, detections = detector.detect_image(test_image_path)
    print(f"✓ Loaded image shape: {image.shape} (H x W x C)")
    print(f"✓ Raw detections count: {len(detections)}")

    # 4. Vehicle Class Filtering Verification
    print("\n[Step 4] Verifying Vehicle Class Filtering...")
    valid_classes = {"car", "motorcycle", "bus", "truck"}
    all_classes_valid = len(detections) > 0 and all(
        det["class_name"] in valid_classes for det in detections
    )
    if all_classes_valid:
        print(f"✓ All detected classes strictly belong to {valid_classes}")
        results["Vehicle filtering"] = True
    else:
        print(f"✗ Detected classes violate target filtering: {[d['class_name'] for d in detections]}")
        results["Vehicle filtering"] = False

    # 5. Coordinate, Confidence, Center & Bottom Point Extractions
    print("\n[Step 5] Validating Detection Fields & Coordinate Math...")
    boxes_valid = True
    conf_valid = True
    center_valid = True
    bottom_valid = True

    for idx, det in enumerate(detections, start=1):
        cls_name = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        bcx, bcy = det["bottom_center"]

        print(
            f"  Detection #{idx}: "
            f"Class={cls_name.upper()} | "
            f"Conf={conf:.3f} | "
            f"Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | "
            f"Center=({cx}, {cy}) | "
            f"BottomGround=({bcx}, {bcy})"
        )

        # Validate bounding box coordinates
        if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
            boxes_valid = False

        # Validate confidence score bounds
        if not (0.0 <= conf <= 1.0):
            conf_valid = False

        # Validate center point calculation
        expected_cx = int(round((x1 + x2) / 2.0))
        expected_cy = int(round((y1 + y2) / 2.0))
        if (cx, cy) != (expected_cx, expected_cy):
            center_valid = False

        # Validate bottom ground-contact point calculation
        expected_bcx = expected_cx
        expected_bcy = int(round(y2))
        if (bcx, bcy) != (expected_bcx, expected_bcy):
            bottom_valid = False

    results["Bounding boxes"] = boxes_valid
    results["Confidence extraction"] = conf_valid
    results["Center point calculation"] = center_valid
    results["Bottom point calculation"] = bottom_valid

    # 6. Annotation Rendering
    print("\n[Step 6] Generating Visual Annotations...")
    try:
        annotated_image = detector.draw_detections(
            image, detections, draw_center=True, draw_bottom_center=True
        )
        if annotated_image is not None and annotated_image.shape == image.shape:
            print("✓ Visual annotation overlay successfully generated.")
            results["Annotation"] = True
        else:
            results["Annotation"] = False
            print("✗ Annotation failed: output array mismatch.")
    except Exception as exc:
        results["Annotation"] = False
        print(f"✗ Annotation failed: {exc}")

    # 7. Output File Generation & Verification
    print("\n[Step 7] Saving Annotated Image & Verifying Integrity...")
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase1_detector_test.jpg"

    try:
        save_success = cv2.imwrite(str(output_path), annotated_image)
        if not save_success or not output_path.is_file() or output_path.stat().st_size == 0:
            print(f"✗ Failed to save output image to {output_path}")
            results["Output generation"] = False
        else:
            # Verify file can be reopened and decoded
            reopened = cv2.imread(str(output_path))
            if reopened is not None and reopened.shape == image.shape:
                print(f"✓ Saved and verified output image: {output_path} ({output_path.stat().st_size} bytes)")
                results["Output generation"] = True
            else:
                print("✗ Reopening saved image failed or corrupted.")
                results["Output generation"] = False
    except Exception as exc:
        results["Output generation"] = False
        print(f"✗ Output generation encountered exception: {exc}")

    # 8. Edge Case Validations
    print("\n[Step 8] Testing Edge Cases...")
    edge_cases_passed = True

    # Edge Case 1: Missing input file
    try:
        detector.detect_image(PROJECT_ROOT / "assets" / "non_existent_image_12345.jpg")
        print("✗ Missing file edge case FAILED: Expected FileNotFoundError")
        edge_cases_passed = False
    except FileNotFoundError:
        print("✓ Edge Case (Missing input file): Raised FileNotFoundError as expected.")

    # Edge Case 2: Blank image (empty/no detections)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_detections = detector.detect(blank_frame)
    if len(blank_detections) == 0:
        print("✓ Edge Case (Blank image / No detections): Handled cleanly, returned empty list.")
    else:
        print(f"✗ Blank image returned unexpected detections: {blank_detections}")
        edge_cases_passed = False

    # Edge Case 3: Explicit CPU Execution
    try:
        cpu_detector = VehicleDetector(model_path=str(model_path), device="cpu")
        cpu_dets = cpu_detector.detect(image)
        if len(cpu_dets) > 0 and str(cpu_detector.device).lower() == "cpu":
            print("✓ Edge Case (Explicit CPU Execution): Successfully ran inference on CPU.")
        else:
            print("✗ Explicit CPU execution failed.")
            edge_cases_passed = False
    except Exception as exc:
        print(f"✗ CPU execution failed with error: {exc}")
        edge_cases_passed = False

    # Edge Case 4: Malformed input (None, 1D array, empty array)
    try:
        assert detector.detect(None) == []  # type: ignore
        assert detector.detect(np.zeros((0,), dtype=np.uint8)) == []
        assert detector.detect(np.zeros((100,), dtype=np.uint8)) == []
        print("✓ Edge Case (Malformed inputs): Handled safely without crashes.")
    except Exception as exc:
        print(f"✗ Malformed inputs caused an exception: {exc}")
        edge_cases_passed = False

    # Summary Report
    all_passed = all(results.values()) and edge_cases_passed
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION")
    print("=" * 60)
    for key, passed in results.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")
    print(f"Edge cases: {'PASS' if edge_cases_passed else 'FAIL'}")
    print("\nOverall Phase 1:")
    print("PASS" if all_passed else "FAIL")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_phase1_verification()
    sys.exit(0 if success else 1)
