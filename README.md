# Smart Traffic Vehicle Detection, Tracking & Counting System

**Author:** Darshan S.  
**Tech Stack:** Python, Ultralytics YOLO11, OpenCV, ByteTrack, BoT-SORT, Pandas, Streamlit

---

## 📌 Project Overview
A modular, production-grade computer vision application designed to detect, track, and count vehicles in traffic surveillance videos. The system supports multi-class vehicle identification (cars, buses, trucks, motorcycles), assigns persistent track IDs, performs trajectory smoothing, tracks line crossings bidirectionally (UP / DOWN), prevents duplicate counting, and generates comparative tracking analytics.

---

## 🗂️ Project Structure

```text
smart-traffic-vehicle-counter/
├── app/
│   └── streamlit_app.py        # Streamlit interactive web dashboard
├── assets/                     # Demo assets, sample screenshots, diagrams
├── configs/
│   ├── botsort.yaml            # BoT-SORT tracker configuration
│   └── bytetrack.yaml          # ByteTrack tracker configuration
├── datasets/                   # Test videos and evaluation data
├── models/                     # YOLO11 model weights (.pt files)
├── notebooks/
│   └── vehicle_counting.ipynb  # Interactive development & experimentation notebook
├── outputs/                    # Annotated videos, tracking logs, and CSV reports
├── src/
│   ├── __init__.py             # Package initialization
│   ├── counter.py              # Bidirectional line-crossing & counting logic
│   ├── detector.py             # YOLO11 inference wrapper & vehicle class filtering
│   ├── metrics.py              # Performance benchmarks (FPS, ID switches, comparisons)
│   ├── tracker.py              # ByteTrack & BoT-SORT tracker integrations
│   └── video_processor.py      # Video I/O pipeline and visual annotation renderer
├── .gitignore
├── README.md
└── requirements.txt            # Project dependencies
```

---

## 🚗 Target Vehicle Classes
- `car` (COCO class ID: 2)
- `motorcycle` (COCO class ID: 3)
- `bus` (COCO class ID: 5)
- `truck` (COCO class ID: 7)

---

## 🚀 Installation & Setup

1. **Activate Virtual Environment:**
   ```bash
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
