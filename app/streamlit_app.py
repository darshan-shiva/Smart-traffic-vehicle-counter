from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple
import streamlit as st
import pandas as pd
import streamlit as st

# ============================================================
# Smart Traffic Vehicle Counter - Phase 6 Streamlit Dashboard
# ============================================================
# Expected structure:
# project/
#   app/streamlit_app.py
#   run_pipeline.py
#   models/yolo11n.pt
#   data/
#   outputs/
#   src/
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "streamlit_uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PIPELINE_FILE = PROJECT_ROOT / "run_pipeline.py"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Smart Traffic Vehicle Counter",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero { padding:28px 30px; border-radius:18px; margin-bottom:24px;
        border:1px solid rgba(128,128,128,.25); background:rgba(255,255,255,.015); }
.hero h1 { margin:0; font-size:2.35rem; line-height:1.2; }
.hero p { margin:10px 0 0; opacity:.72; font-size:1rem; }
.metric-card { padding:20px 22px; border-radius:16px; border:1px solid rgba(128,128,128,.25);
               min-height:115px; background:rgba(255,255,255,.01); }
.metric-label { font-size:.76rem; opacity:.68; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
.metric-value { font-size:2.15rem; line-height:1; font-weight:750; }
.section-title { font-size:1.45rem; font-weight:700; margin-top:8px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "analysis_complete": False,
    "analysis_failed": False,
    "output_video": None,
    "browser_video": None,
    "metrics_file": None,
    "metrics": None,
    "pipeline_log": "",
    "processing_time": 0.0,
    "input_file": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def save_uploaded_video(uploaded_file) -> Optional[Path]:
    if uploaded_file is None:
        return None
    safe_name = Path(uploaded_file.name).name
    destination = UPLOAD_DIR / safe_name
    with open(destination, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return destination


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def first_metric(metrics: Optional[Dict[str, Any]], *keys: str, default: Any = 0) -> Any:
    if not isinstance(metrics, dict):
        return default
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return default


def parse_time_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else 0.0


def tracker_output_paths(tracker: str) -> Tuple[Path, Path]:
    return (
        OUTPUT_DIR / f"phase6_{tracker}.mp4",
        OUTPUT_DIR / f"phase6_{tracker}_metrics.json",
    )


def locate_phase6_outputs(tracker: str) -> Tuple[Optional[Path], Optional[Path]]:
    video, metrics = tracker_output_paths(tracker)
    return (video if video.exists() else None,
            metrics if metrics.exists() else None)


def run_pipeline(input_file: Path, tracker: str, max_frames: int):
    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(f"Pipeline file was not found:\n{PIPELINE_FILE}")

    command = [
        sys.executable, str(PIPELINE_FILE),
        "--input", str(input_file),
        "--tracker", tracker,
        "--max-frames", str(max_frames),
        "--line-ratio", "0.65",
    ]
    started = time.time()
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    log = stdout + ("\n\n--- STDERR ---\n" + stderr if stderr else "")
    return result, elapsed, log


def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg from PATH or a WinGet Gyan.FFmpeg installation."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    winget_root = (
        Path.home()
        / "AppData"
        / "Local"
        / "Microsoft"
        / "WinGet"
        / "Packages"
    )

    if winget_root.exists():
        matches = sorted(winget_root.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"))
        if matches:
            return str(matches[-1])

    return None


def make_browser_compatible_video(source_video: Optional[Path], tracker: str) -> Optional[Path]:
    """Convert OpenCV MP4 output to browser-compatible H.264 MP4."""
    if source_video is None or not source_video.exists():
        return None

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return source_video

    browser_video = OUTPUT_DIR / f"streamlit_{tracker}_h264.mp4"
    try:
        if (browser_video.exists() and
                browser_video.stat().st_mtime >= source_video.stat().st_mtime):
            return browser_video

        command = [
            ffmpeg, "-y", "-i", str(source_video),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
            str(browser_video),
        ]
        result = subprocess.run(
            command, cwd=str(PROJECT_ROOT), capture_output=True,
            text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0 and browser_video.exists():
            return browser_video
    except Exception:
        pass
    return source_video


def format_int(value: Any) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def format_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.00"


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🚦 Smart Traffic Vehicle Counter</h1>
    <p>YOLO11 Detection · Multi-Object Tracking · Vehicle Counting ·
    Speed Estimation · Traffic Analytics</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    tracker = st.selectbox(
        "Tracker", ["bytetrack", "botsort"], index=0,
        format_func=lambda value: value.upper()
    )
    max_frames = st.number_input(
        "Maximum Frames", min_value=1, max_value=10000,
        value=401, step=1
    )
    st.divider()
    st.markdown("### Processing Pipeline")
    st.markdown("""
    **1.** Upload video  
    **2.** YOLO11 detection  
    **3.** Object tracking  
    **4.** Track management  
    **5.** Counting-line crossing 
    **6.** Speed estimation  
    **7.** Metrics generation  
    **8.** Result visualization
    """)
    st.divider()
    if st.button("🔄 Reset Results", use_container_width=True):
        reset_results()
        st.rerun()


# ============================================================
# INPUT VIDEO
# ============================================================

st.markdown('<div class="section-title">🎥 Input Traffic Video</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv", "mpeg4"],
)

if uploaded is not None:
    st.success(f"Loaded: {uploaded.name}")
    with st.expander("Preview Input Video", expanded=False):
        st.video(uploaded)


# ============================================================
# RUN ANALYSIS
# ============================================================

st.markdown('<div class="section-title">▶️ Run Analysis</div>', unsafe_allow_html=True)
left, right = st.columns([3, 1])
with left:
    st.caption(f"Tracker: **{tracker.upper()}**  |  Maximum frames: **{int(max_frames)}**")
with right:
    start_analysis = st.button(
        "🚀 Start Analysis", type="primary", use_container_width=True,
        disabled=uploaded is None
    )

if start_analysis:
    reset_results()
    input_file = save_uploaded_video(uploaded)
    if input_file is None:
        st.error("Could not save the uploaded video.")
        st.stop()
    st.session_state.input_file = str(input_file)

    progress = st.progress(0)
    status = st.empty()
    status.info("Preparing video...")
    progress.progress(5)

    # Remove only the browser-conversion file for this tracker.
    stale_browser = OUTPUT_DIR / f"streamlit_{tracker}_h264.mp4"
    try:
        if stale_browser.exists():
            stale_browser.unlink()
    except OSError:
        pass

    status.info(f"Running YOLO11 + {tracker.upper()} on {int(max_frames)} frames...")
    progress.progress(10)

    try:
        result, elapsed, log = run_pipeline(input_file, tracker, int(max_frames))
        st.session_state.processing_time = elapsed
        st.session_state.pipeline_log = log
        progress.progress(90)

        if result.returncode != 0:
            st.session_state.analysis_failed = True
            progress.progress(100)
            status.error("Pipeline execution failed.")
            with st.expander("❌ View Pipeline Error", expanded=True):
                st.code(log or "No error output was returned.", language="text")
            st.stop()

        status.info("Reading Phase 6 results...")
        output_video, metrics_file = locate_phase6_outputs(tracker)
        metrics = load_json(metrics_file)

        if output_video is not None:
            status.info("Converting processed video to browser-compatible H.264...")
            browser_video = make_browser_compatible_video(output_video, tracker)
        else:
            browser_video = None

        st.session_state.output_video = str(output_video) if output_video else None
        st.session_state.browser_video = str(browser_video) if browser_video else None
        st.session_state.metrics_file = str(metrics_file) if metrics_file else None
        st.session_state.metrics = metrics
        st.session_state.analysis_complete = True
        progress.progress(100)

        if metrics is not None and output_video is not None:
            status.success(f"Traffic analysis completed in {elapsed:.2f} seconds.")
        elif metrics is not None:
            status.warning("Analysis completed, but the processed video was not found.")
        else:
            status.warning("Pipeline completed, but the metrics JSON could not be read.")

    except Exception as exc:
        st.session_state.analysis_failed = True
        st.session_state.analysis_complete = False
        progress.progress(100)
        status.error("Unexpected error during analysis.")
        with st.expander("❌ Technical Error", expanded=True):
            st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_complete:
    st.divider()
    st.markdown('<div class="section-title">📊 Traffic Analysis Results</div>', unsafe_allow_html=True)

    metrics = st.session_state.metrics

    if metrics:
        # IMPORTANT: metrics.py uses these exact keys.
        # The previous dashboard searched for snake_case keys and therefore
        # displayed 0 even when the pipeline had produced real counts.
        up = first_metric(metrics, "UP count", "up_count", "UP", default=0)
        down = first_metric(metrics, "DOWN count", "down_count", "DOWN", default=0)
        total = first_metric(metrics, "TOTAL count", "total_count", "TOTAL", default=None)
        unique = first_metric(metrics, "Unique Track IDs", "unique_track_ids", "unique_tracks", default=0)
        avg_length = first_metric(metrics, "Average track length", "average_track_length", "avg_track_length", default=0)
        fps = first_metric(metrics, "Processing FPS", "processing_fps", "fps", default=0)
        metric_time = first_metric(metrics, "Processing time", "processing_time", default=None)

        if total is None:
            try:
                total = float(up) + float(down)
            except (TypeError, ValueError):
                total = 0

        display_time = parse_time_seconds(metric_time) if metric_time is not None else st.session_state.processing_time

        st.markdown("#### 🚦 Traffic Overview")

        a, b, c, d = st.columns(4)
        cards = [
            (a, "Total Vehicles", format_int(total)),
            (b, "Down Count", format_int(down)),
            (c, "Up Count", format_int(up)),
            (d, "Unique Tracks", format_int(unique)),
        ]
        for column, label, value in cards:
            with column:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 📈 Traffic Analytics")

        st.caption("Vehicle flow detected across the counting line.")
        st.bar_chart(
            {
                "DOWN": [float(down)],
                "UP": [float(up)],
            },
            height=240,
        )

        st.markdown("#### 🔍 Tracking & Processing Summary")
        x, y, z = st.columns(3)
        with x:
            st.metric("Average Track Length", format_float(avg_length))
        with y:
            st.metric("Processing FPS", format_float(fps))
        with z:
            st.metric("Processing Time", f"{display_time:.2f} s")
                    # ========================================================
        # VEHICLE CLASS BREAKDOWN
        # ========================================================

        st.markdown(
            '<div class="section-title">🚗 Vehicle Class Breakdown</div>',
            unsafe_allow_html=True
        )

        # Read class-level counting data from metrics JSON
        class_counts = first_metric(
            metrics,
            "Vehicle Class Counts",
            "vehicle_class_counts",
            default={}
        )

        direction_class_counts = first_metric(
            metrics,
            "Vehicle Class Direction Counts",
            "vehicle_class_direction_counts",
            default={}
        )

        if not isinstance(class_counts, dict):
            class_counts = {}

        # Normalize values
        car_count = int(class_counts.get("car", 0) or 0)
        truck_count = int(class_counts.get("truck", 0) or 0)
        bus_count = int(class_counts.get("bus", 0) or 0)
        motorcycle_count = int(class_counts.get("motorcycle", 0) or 0)

        # --------------------------------------------------------
        # CLASS METRIC CARDS
        # --------------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        vehicle_cards = [
            (c1, "🚗 Cars", car_count),
            (c2, "🚚 Trucks", truck_count),
            (c3, "🚌 Buses", bus_count),
            (c4, "🏍️ Motorcycles", motorcycle_count),
        ]

        for column, label, value in vehicle_cards:
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        # --------------------------------------------------------
        # VEHICLE CLASS CHART + TABLE
        # --------------------------------------------------------

        chart_col, table_col = st.columns([2, 1])

        class_data = pd.DataFrame({
            "Vehicle": [
                "Cars",
                "Trucks",
                "Buses",
                "Motorcycles"
            ],
            "Count": [
                car_count,
                truck_count,
                bus_count,
                motorcycle_count
            ]
        })

        with chart_col:
            st.markdown("#### 📊 Vehicle Distribution")

            chart_data = class_data.set_index("Vehicle")
            st.bar_chart(
                chart_data,
                y="Count",
                use_container_width=True
            )

        with table_col:
            st.markdown("#### 📋 Count Summary")

            display_table = class_data.copy()
            display_table["Percentage"] = (
                display_table["Count"] / max(car_count + truck_count + bus_count + motorcycle_count, 1) * 100
            ).round(1)

            display_table["Percentage"] = (
                display_table["Percentage"].astype(str) + "%"
            )

            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True
            )

        # --------------------------------------------------------
        # DIRECTION BY VEHICLE CLASS
        # --------------------------------------------------------

        st.markdown("#### ↕️ Vehicle Direction Breakdown")

        direction_rows = []

        for vehicle in ["car", "truck", "bus", "motorcycle"]:
            data = direction_class_counts.get(vehicle, {})

            if not isinstance(data, dict):
                data = {}

            direction_rows.append({
                "Vehicle": vehicle.title(),
                "UP": int(data.get("up", 0) or 0),
                "DOWN": int(data.get("down", 0) or 0),
                "Total": (
                    int(data.get("up", 0) or 0)
                    + int(data.get("down", 0) or 0)
                )
            })

        direction_df = pd.DataFrame(direction_rows)

        st.dataframe(
            direction_df,
            hide_index=True,
            use_container_width=True
        )

        # --------------------------------------------------------
        # ANALYSIS SUMMARY
        # --------------------------------------------------------

        total_classified = (
            car_count
            + truck_count
            + bus_count
            + motorcycle_count
        )

        if total_classified > 0:
            dominant_vehicle = max(
                {
                    "Cars": car_count,
                    "Trucks": truck_count,
                    "Buses": bus_count,
                    "Motorcycles": motorcycle_count,
                },
                key=lambda x: {
                    "Cars": car_count,
                    "Trucks": truck_count,
                    "Buses": bus_count,
                    "Motorcycles": motorcycle_count,
                }[x]
            )

            dominant_count = {
                "Cars": car_count,
                "Trucks": truck_count,
                "Buses": bus_count,
                "Motorcycles": motorcycle_count,
            }[dominant_vehicle]

            dominant_percentage = (
                dominant_count / total_classified
            ) * 100

            st.info(
                f"🚦 **{total_classified:,} vehicles** crossed the counting line. "
                f"**{dominant_vehicle}** were the largest vehicle category "
                f"with **{dominant_count:,} ({dominant_percentage:.1f}%)** "
                f"of classified vehicles."
            )
        else:
            st.info("No vehicle-class crossing data was recorded.")


        # Show vehicle-class analytics only if the metrics JSON actually
        # contains class-level information. No values are invented.
        class_counts = first_metric(
            metrics,
            "Vehicle Class Counts",
            "vehicle_class_counts",
            "class_counts",
            "vehicle_counts",
            default=None,
        )

        if isinstance(class_counts, dict) and class_counts:
            st.markdown("#### 🚗 Vehicle Class Distribution")

            normalized_classes = {}
            for class_name, count in class_counts.items():
                try:
                    normalized_classes[str(class_name).upper()] = int(float(count))
                except (TypeError, ValueError):
                    continue

            if normalized_classes:
                class_df = pd.DataFrame(
                    {
                        "Vehicle Class": list(normalized_classes.keys()),
                        "Count": list(normalized_classes.values()),
                    }
                ).set_index("Vehicle Class")

                st.bar_chart(class_df, height=260)

                # Directional class breakdown, when supplied by the pipeline.
                direction_counts = first_metric(
                    metrics,
                    "Vehicle Class Direction Counts",
                    "vehicle_class_direction_counts",
                    default=None,
                )

                if isinstance(direction_counts, dict):
                    rows = []
                    for class_name, directions in direction_counts.items():
                        if not isinstance(directions, dict):
                            continue
                        try:
                            rows.append(
                                {
                                    "Vehicle Class": str(class_name).upper(),
                                    "UP": int(directions.get("up", 0)),
                                    "DOWN": int(directions.get("down", 0)),
                                    "TOTAL": int(directions.get("up", 0))
                                    + int(directions.get("down", 0)),
                                }
                            )
                        except (TypeError, ValueError):
                            continue

                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows).set_index("Vehicle Class"),
                            use_container_width=True,
                        )

        st.markdown("#### 📊 Analysis Summary")
        if float(total) > 0:
            down_pct = (float(down) / float(total)) * 100
            up_pct = (float(up) / float(total)) * 100

            st.info(
                f"**{format_int(total)} vehicles** crossed the counting line. "
                f"**{down_pct:.1f}%** were recorded DOWN and "
                f"**{up_pct:.1f}%** were recorded UP."
            )
        else:
            st.info("No counting-line crossings were recorded in this analysis.")
    else:
        st.warning("The pipeline completed, but no readable metrics JSON was found.")

        # ========================================================
        # SPEED ANALYTICS
        # ========================================================
        if metrics:
            speed_estimates = first_metric(
                metrics,
                "Speed Estimates",
                "speed_estimates",
                default=0,
            )

            average_speed = first_metric(
                metrics,
                "Average Speed (km/h)",
                "average_speed_kmh",
                "average_speed",
                default=0,
            )

            maximum_speed = first_metric(
                metrics,
                "Maximum Speed (km/h)",
                "maximum_speed_kmh",
                "maximum_speed",
                default=0,
            )

            speed_distance = first_metric(
                metrics,
                "Speed Distance (m)",
                "speed_distance_m",
                "speed_distance",
                default=0,
            )

            speed_records = first_metric(
                metrics,
                "Speed Records",
                "speed_records",
                default=[],
            )

            if not isinstance(speed_records, list):
                speed_records = []

            st.markdown(
                '<div class="section-title">🏎️ Speed Analytics</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Estimated vehicle speed from the time taken to travel "
                "between the two speed measurement lines."
            )

            # ----------------------------------------------------
            # SUMMARY CARDS
            # ----------------------------------------------------
            s1, s2, s3, s4 = st.columns(4)

            speed_cards = [
                (s1, "Speed Estimates", format_int(speed_estimates)),
                (s2, "Average Speed", f"{format_float(average_speed)} km/h"),
                (s3, "Maximum Speed", f"{format_float(maximum_speed)} km/h"),
                (s4, "Line Distance", f"{format_float(speed_distance)} m"),
            ]

            for column, label, value in speed_cards:
                with column:
                    st.metric(label, value)

            # ----------------------------------------------------
            # INDIVIDUAL VEHICLE SPEEDS
            # ----------------------------------------------------
            st.markdown("### 🚗 Individual Vehicle Speeds")

            if speed_records:

                speed_table = []

                for record in speed_records:
                    try:
                        track_id = int(record.get("track_id", 0))
                        speed_kmh = float(record.get("speed_kmh", 0))
                        elapsed_seconds = float(
                            record.get("elapsed_seconds", 0)
                        )

                        speed_table.append(
                            {
                                "Track ID": track_id,
                                "Speed (km/h)": round(speed_kmh, 2),
                                "Travel Time (s)": round(
                                    elapsed_seconds,
                                    3,
                                ),
                            }
                        )

                    except (TypeError, ValueError):
                        continue

                if speed_table:

                    # Sort by speed, highest first
                    speed_table = sorted(
                        speed_table,
                        key=lambda x: x["Speed (km/h)"],
                        reverse=True,
                    )

                    st.dataframe(
                        speed_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.warning(
                        "Speed records were found, but they could not "
                        "be displayed."
                    )

            else:
                st.info(
                    "No individual speed records are available."
                )

            # ----------------------------------------------------
            # SPEED DISTRIBUTION
            # ----------------------------------------------------
            st.markdown("### 📊 Speed Distribution")

            if speed_records:

                speed_values = []

                for record in speed_records:
                    try:
                        speed_values.append(
                            float(record.get("speed_kmh", 0))
                        )
                    except (TypeError, ValueError):
                        pass

                if speed_values:

                    # Speed ranges
                    bins = [
                        0,
                        5,
                        10,
                        15,
                        20,
                        25,
                        30,
                        40,
                        50,
                        float("inf"),
                    ]

                    labels = [
                        "0–5",
                        "5–10",
                        "10–15",
                        "15–20",
                        "20–25",
                        "25–30",
                        "30–40",
                        "40–50",
                        "50+",
                    ]

                    speed_series = pd.Series(speed_values)

                    distribution = (
                        pd.cut(
                            speed_series,
                            bins=bins,
                            labels=labels,
                            right=False,
                        )
                        .value_counts()
                        .reindex(labels, fill_value=0)
                    )

                    distribution_df = pd.DataFrame(
                        {
                            "Speed Range (km/h)": distribution.index,
                            "Vehicles": distribution.values,
                        }
                    )

                    st.bar_chart(
                        distribution_df.set_index(
                            "Speed Range (km/h)"
                        )
                    )

                else:
                    st.info(
                        "No valid speed values available "
                        "for the distribution."
                    )

            # ----------------------------------------------------
            # SPEED STATISTICS
            # ----------------------------------------------------
            if speed_records and speed_values:

                min_speed = min(speed_values)
                avg_speed = sum(speed_values) / len(speed_values)
                max_speed = max(speed_values)

                st.markdown("### 📈 Speed Statistics")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Minimum Speed",
                        f"{min_speed:.2f} km/h",
                    )

                with c2:
                    st.metric(
                        "Average Speed",
                        f"{avg_speed:.2f} km/h",
                    )

                with c3:
                    st.metric(
                        "Maximum Speed",
                        f"{max_speed:.2f} km/h",
                    )

            # ----------------------------------------------------
            # CALIBRATION INFO
            # ----------------------------------------------------
            st.info(
                f"📏 Speed is calculated using a calibrated distance "
                f"of {format_float(speed_distance)} m between "
                f"Speed Line 1 and Speed Line 2. "
                f"The values are estimates and depend on scene "
                f"calibration and tracking accuracy."
            )

    # ========================================================
    # PROCESSED VIDEO
    # ========================================================
    st.markdown('<div class="section-title">🎬 Processed Video</div>', unsafe_allow_html=True)

    browser_video = st.session_state.browser_video
    output_video = st.session_state.output_video

    if browser_video and Path(browser_video).exists():
        browser_path = Path(browser_video)
        browser_bytes = browser_path.read_bytes()

        st.video(browser_bytes, format="video/mp4")

        st.download_button(
            "⬇️ Download Processed Video",
            data=browser_bytes,
            file_name=f"phase6_{tracker}_processed.mp4",
            mime="video/mp4",
        )
    elif output_video and Path(output_video).exists():
        st.warning("The processed video exists, but browser-compatible H.264 conversion was unavailable.")
        with open(output_video, "rb") as file:
            st.download_button(
                "⬇️ Download Original Processed Video", data=file,
                file_name=Path(output_video).name, mime="video/mp4"
            )
    else:
        st.warning("Processed video could not be located.")

    # ========================================================
    # DETAILS
    # ========================================================
    if metrics:
        with st.expander("📄 Detailed Metrics", expanded=False):
            st.json(metrics)
            if st.session_state.metrics_file and Path(st.session_state.metrics_file).exists():
                with open(st.session_state.metrics_file, "rb") as file:
                    st.download_button(
                        "⬇️ Download Metrics JSON", data=file,
                        file_name=Path(st.session_state.metrics_file).name,
                        mime="application/json"
                    )

    if st.session_state.pipeline_log:
        with st.expander("🧾 Pipeline Log", expanded=False):
            st.code(st.session_state.pipeline_log, language="text")

    with st.expander("📁 Output Files", expanded=False):
        if output_video:
            st.write(f"**Original output:** `{output_video}`")
        if browser_video:
            st.write(f"**Browser video:** `{browser_video}`")
        if st.session_state.metrics_file:
            st.write(f"**Metrics JSON:** `{st.session_state.metrics_file}`")
else:
    st.info("Upload a traffic video, choose the tracker and frame count, then click **Start Analysis**.")

st.caption("Smart Traffic Vehicle Detection, Tracking & Counting System | YOLO11 + ByteTrack/BoT-SORT + Streamlit")
