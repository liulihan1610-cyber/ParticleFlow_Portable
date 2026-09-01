import hashlib
import os
import tempfile
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.colors import qualitative
except ImportError:
    st.error(
        "This version needs Plotly for the interactive chart. "
        "Install it in Terminal, then restart the app."
    )
    st.code("python3 -m pip install plotly")
    st.stop()

try:
    from streamlit_cropper import st_cropper
except ImportError:
    st.error(
        "This version needs the streamlit-cropper component. "
        "Install it in Terminal, then restart the app."
    )
    st.code("python3 -m pip install streamlit-cropper")
    st.stop()


# ==================================================
# Page setup
# ==================================================
st.set_page_config(
    page_title="Particle Flow Video Analysis Tool",
    layout="wide",
)

st.title("Particle Flow Video Analysis Tool")
st.write(
    "Follow the five steps below to prepare the video, define the analysis "
    "regions, run the analysis and download the results."
)

st.markdown(
    """
    <style>
    .workflow-wrap {
        display: flex;
        align-items: stretch;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.7rem 0 1.1rem 0;
    }
    .workflow-step {
        flex: 1 1 145px;
        min-width: 135px;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 0.75rem;
        padding: 0.75rem 0.8rem;
        background: rgba(240, 242, 246, 0.55);
    }
    .workflow-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.65rem;
        height: 1.65rem;
        border-radius: 50%;
        background: #2e86de;
        color: white;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .workflow-title {
        display: block;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .workflow-detail {
        display: block;
        font-size: 0.88rem;
        color: rgba(49, 51, 63, 0.72);
        line-height: 1.25;
    }
    </style>
    <div class="workflow-wrap">
        <div class="workflow-step">
            <span class="workflow-number">1</span>
            <span class="workflow-title">Upload video</span>
            <span class="workflow-detail">Select the experimental video.</span>
        </div>
        <div class="workflow-step">
            <span class="workflow-number">2</span>
            <span class="workflow-title">Straighten and crop</span>
            <span class="workflow-detail">Align the tube and keep the useful area.</span>
        </div>
        <div class="workflow-step">
            <span class="workflow-number">3</span>
            <span class="workflow-title">Select reference ROI</span>
            <span class="workflow-detail">Mark the stationary baseline region at the tube bottom.</span>
        </div>
        <div class="workflow-step">
            <span class="workflow-number">4</span>
            <span class="workflow-title">Select analysis area</span>
            <span class="workflow-detail">Draw the region above the reference and choose 1–100 zones.</span>
        </div>
        <div class="workflow-step">
            <span class="workflow-number">5</span>
            <span class="workflow-title">Analyse and download</span>
            <span class="workflow-detail">Review the plots and export the results.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

REFERENCE_ROI_LABEL = "Reference ROI"


# ==================================================
# Session-state helpers
# ==================================================
def initialise_state() -> None:
    defaults = {
        "uploaded_video_id": None,
        "uploaded_filename": None,
        "video_path": None,
        "rotation_angle": 0.0,
        "crop_video_enabled": True,
        "video_crop_box": None,
        "video_crop_pending": None,
        "crop_applied": False,
        "crop_revision": 0,
        "main_area_box": None,
        "main_area_pending": None,
        "main_area_applied": False,
        "reference_roi_box": None,
        "reference_roi_pending": None,
        "reference_roi_applied": False,
        "reference_revision": 0,
        "reference_top": None,
        "zone_count": 5,
        "use_reference_roi": True,
        "preview_size": "Medium",
        "setup_preview_frame_number": 0,
        "processed_signature": None,
        "analysis_results": None,
        "workflow_step": 1,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_state()


def safe_remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def reset_for_new_video(
    video_id: str,
    filename: str,
    video_path: str,
) -> None:
    old_video_path = st.session_state.get("video_path")
    if old_video_path != video_path:
        safe_remove(old_video_path)

    old_results = st.session_state.get("analysis_results")
    if old_results:
        safe_remove(old_results.get("processed_video_path"))

    st.session_state.uploaded_video_id = video_id
    st.session_state.uploaded_filename = filename
    st.session_state.video_path = video_path
    st.session_state.rotation_angle = 0.0
    st.session_state.crop_video_enabled = True
    st.session_state.video_crop_box = None
    st.session_state.video_crop_pending = None
    st.session_state.crop_applied = False
    st.session_state.crop_revision = 0
    st.session_state.main_area_box = None
    st.session_state.main_area_pending = None
    st.session_state.main_area_applied = False
    st.session_state.reference_roi_box = None
    st.session_state.reference_roi_pending = None
    st.session_state.reference_roi_applied = False
    st.session_state.reference_revision = 0
    st.session_state.reference_top = None
    st.session_state.zone_count = 5
    st.session_state.use_reference_roi = True
    st.session_state.preview_size = "Medium"
    st.session_state.setup_preview_frame_number = 0
    st.session_state.processed_signature = None
    st.session_state.analysis_results = None
    st.session_state.workflow_step = 1


def clear_analysis_results() -> None:
    old_results = st.session_state.get("analysis_results")
    if old_results:
        safe_remove(old_results.get("processed_video_path"))
    st.session_state.analysis_results = None


# ==================================================
# Video and image helpers
# ==================================================
def save_uploaded_video(uploaded_file) -> Tuple[str, str]:
    video_bytes = uploaded_file.getvalue()
    video_id = hashlib.sha1(video_bytes).hexdigest()[:16]
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(video_bytes)
        video_path = temp_file.name

    return video_id, video_path


def get_video_info(video_path: str) -> Optional[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "duration": duration,
    }


def get_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    success, frame = cap.read()
    cap.release()
    return frame if success else None


def rotate_frame_bound(frame: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a frame without cutting off its corners."""
    if abs(angle) < 1e-9:
        return frame.copy()

    height, width = frame.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int((height * sine) + (width * cosine))
    new_height = int((height * cosine) + (width * sine))

    matrix[0, 2] += (new_width / 2.0) - centre[0]
    matrix[1, 2] += (new_height / 2.0) - centre[1]

    return cv2.warpAffine(
        frame,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def full_frame_box(frame: np.ndarray) -> dict:
    height, width = frame.shape[:2]
    return {
        "left": 0,
        "top": 0,
        "width": int(width),
        "height": int(height),
    }


def default_crop_box(frame: np.ndarray) -> dict:
    """Return a visible, medium-sized centred crop box for first-time setup."""
    height, width = frame.shape[:2]
    crop_width = min(width, max(40, int(round(width * 0.60))))
    crop_height = min(height, max(40, int(round(height * 0.75))))
    return {
        "left": max(0, int((width - crop_width) / 2)),
        "top": max(0, int((height - crop_height) / 2)),
        "width": int(crop_width),
        "height": int(crop_height),
    }


def normalise_box(
    box: Optional[dict],
    frame_width: int,
    frame_height: int,
    minimum_size: int = 2,
) -> dict:
    if not box:
        return {
            "left": 0,
            "top": 0,
            "width": frame_width,
            "height": frame_height,
        }

    left = int(round(box.get("left", 0)))
    top = int(round(box.get("top", 0)))
    width = int(round(box.get("width", frame_width)))
    height = int(round(box.get("height", frame_height)))

    left = min(max(left, 0), max(frame_width - minimum_size, 0))
    top = min(max(top, 0), max(frame_height - minimum_size, 0))
    width = min(max(width, minimum_size), frame_width - left)
    height = min(max(height, minimum_size), frame_height - top)

    return {
        "left": int(left),
        "top": int(top),
        "width": int(width),
        "height": int(height),
    }


def crop_frame(frame: np.ndarray, crop_box: dict) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    box = normalise_box(crop_box, frame_width, frame_height)
    left = box["left"]
    top = box["top"]
    right = left + box["width"]
    bottom = top + box["height"]
    return frame[top:bottom, left:right].copy()


def preprocess_frame(
    frame: np.ndarray,
    rotation_angle: float,
    crop_box: dict,
) -> np.ndarray:
    rotated = rotate_frame_bound(frame, rotation_angle)
    return crop_frame(rotated, crop_box)


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def box_to_default_coords(box: dict) -> tuple:
    return (
        int(box["left"]),
        int(box["left"] + box["width"]),
        int(box["top"]),
        int(box["top"] + box["height"]),
    )


def default_inner_box(
    frame_width: int,
    frame_height: int,
    horizontal_margin: float = 0.20,
    vertical_margin: float = 0.10,
) -> dict:
    left = int(frame_width * horizontal_margin)
    top = int(frame_height * vertical_margin)
    right = int(frame_width * (1.0 - horizontal_margin))
    bottom = int(frame_height * (1.0 - vertical_margin))
    return {
        "left": left,
        "top": top,
        "width": max(2, right - left),
        "height": max(2, bottom - top),
    }


def default_reference_box(frame_width: int, frame_height: int) -> dict:
    width = max(20, int(frame_width * 0.18))
    height = max(20, int(frame_height * 0.10))
    left = max(0, int((frame_width - width) / 2))
    top = max(0, frame_height - height - int(frame_height * 0.04))
    return {
        "left": left,
        "top": top,
        "width": min(width, frame_width - left),
        "height": min(height, frame_height - top),
    }


PREVIEW_MAX_HEIGHTS = {
    "Compact": 460,
    "Medium": 620,
    "Large": 800,
}

PREVIEW_SIZE_LABELS = {
    "Compact": "Compact",
    "Medium": "Medium (recommended)",
    "Large": "Large",
}


def resize_frame_for_display(
    frame: np.ndarray,
    max_height: int,
    max_width: int = 760,
) -> Tuple[np.ndarray, float, float]:
    """Resize only the on-screen preview and return coordinate scales."""
    original_height, original_width = frame.shape[:2]
    scale = min(
        1.0,
        max_height / max(original_height, 1),
        max_width / max(original_width, 1),
    )

    display_width = max(1, int(round(original_width * scale)))
    display_height = max(1, int(round(original_height * scale)))

    if display_width == original_width and display_height == original_height:
        display_frame = frame.copy()
    else:
        display_frame = cv2.resize(
            frame,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

    scale_x = display_width / max(original_width, 1)
    scale_y = display_height / max(original_height, 1)
    return display_frame, scale_x, scale_y


def scale_box_to_display(box: dict, scale_x: float, scale_y: float) -> dict:
    return {
        "left": int(round(box["left"] * scale_x)),
        "top": int(round(box["top"] * scale_y)),
        "width": max(2, int(round(box["width"] * scale_x))),
        "height": max(2, int(round(box["height"] * scale_y))),
    }


def scale_box_to_original(box: dict, scale_x: float, scale_y: float) -> dict:
    return {
        "left": int(round(box["left"] / max(scale_x, 1e-9))),
        "top": int(round(box["top"] / max(scale_y, 1e-9))),
        "width": max(2, int(round(box["width"] / max(scale_x, 1e-9)))),
        "height": max(2, int(round(box["height"] / max(scale_y, 1e-9)))),
    }


def draw_dashed_line(
    image: np.ndarray,
    start: tuple,
    end: tuple,
    color: tuple,
    thickness: int = 2,
    dash_length: int = 14,
    gap_length: int = 9,
) -> None:
    """Draw a dashed line directly onto a BGR image."""
    x1, y1 = start
    x2, y2 = end
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length <= 0:
        return

    direction_x = (x2 - x1) / length
    direction_y = (y2 - y1) / length
    distance = 0.0

    while distance < length:
        dash_end = min(distance + dash_length, length)
        segment_start = (
            int(round(x1 + direction_x * distance)),
            int(round(y1 + direction_y * distance)),
        )
        segment_end = (
            int(round(x1 + direction_x * dash_end)),
            int(round(y1 + direction_y * dash_end)),
        )
        cv2.line(
            image,
            segment_start,
            segment_end,
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += dash_length + gap_length


def add_vertical_alignment_guide(frame: np.ndarray) -> np.ndarray:
    """Add a fixed central vertical guide to the preparation preview only."""
    preview = frame.copy()
    height, width = preview.shape[:2]
    guide_x = width // 2
    guide_color = (255, 120, 0)

    draw_dashed_line(
        preview,
        (guide_x, 0),
        (guide_x, height - 1),
        guide_color,
        thickness=max(2, int(round(width / 650))),
        dash_length=max(12, int(round(height / 55))),
        gap_length=max(7, int(round(height / 90))),
    )
    draw_readable_label(
        preview,
        "Vertical guide",
        min(guide_x + 8, max(width - 170, 0)),
        8,
        guide_color,
        font_scale=0.55,
    )
    return preview


def get_zone_height(main_box: dict, zone_count: int) -> int:
    zones = generate_zone_rois(main_box, zone_count)
    if not zones:
        return 1
    return int(next(iter(zones.values()))[3])


def make_locked_reference_box(
    main_box: dict,
    zone_count: int,
    frame_height: int,
    reference_top: Optional[int],
) -> dict:
    """Reference width, X and height follow the main zones automatically."""
    reference_height = get_zone_height(main_box, zone_count)
    maximum_top = max(0, frame_height - reference_height)

    if reference_top is None:
        suggested_top = main_box["top"] + main_box["height"] + 8
        reference_top = min(suggested_top, maximum_top)

    reference_top = min(max(int(reference_top), 0), maximum_top)
    return {
        "left": int(main_box["left"]),
        "top": int(reference_top),
        "width": int(main_box["width"]),
        "height": int(reference_height),
    }


def draw_roi_setup_preview(
    frame: np.ndarray,
    main_rois: Dict[str, tuple],
    reference_roi: Dict[str, tuple],
) -> np.ndarray:
    """Show main zones, reference ROI and alignment guides in one image."""
    preview = frame.copy()
    image_height, image_width = preview.shape[:2]

    if main_rois:
        main_values = list(main_rois.values())
        left = main_values[0][0]
        right = left + main_values[0][2]
        top = main_values[0][1]
        bottom = main_values[-1][1] + main_values[-1][3]
        guide_color = (180, 180, 180)

        draw_dashed_line(
            preview,
            (left, 0),
            (left, image_height - 1),
            guide_color,
            thickness=2,
        )
        draw_dashed_line(
            preview,
            (right, 0),
            (right, image_height - 1),
            guide_color,
            thickness=2,
        )
        draw_dashed_line(
            preview,
            (max(0, left - 35), bottom),
            (min(image_width - 1, right + 35), bottom),
            guide_color,
            thickness=2,
        )

    main_color = (0, 255, 0)
    reference_color = (0, 255, 255)

    for name, (x, y, width, height) in main_rois.items():
        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            main_color,
            3,
        )
        label_scale = max(0.82, min(1.08, width / 180.0))
        draw_readable_label(
            preview,
            get_display_label(name),
            x + 4,
            y + 4,
            main_color,
            font_scale=label_scale,
        )

    for name, (x, y, width, height) in reference_roi.items():
        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            reference_color,
            4,
        )
        label_scale = max(0.82, min(1.08, width / 180.0))
        draw_readable_label(
            preview,
            "Reference",
            x + 4,
            y + 4,
            reference_color,
            font_scale=label_scale,
        )

    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def crop_to_roi_context(
    image: np.ndarray,
    main_rois: Dict[str, tuple],
    reference_roi: Dict[str, tuple],
) -> np.ndarray:
    """Crop a setup preview around the selected ROIs for a clear close-up.

    This changes only the on-screen preview. ROI coordinates and video analysis
    continue to use the complete prepared frame.
    """
    roi_values = list(main_rois.values()) + list(reference_roi.values())
    if not roi_values:
        return image.copy()

    image_height, image_width = image.shape[:2]
    left = min(roi[0] for roi in roi_values)
    top = min(roi[1] for roi in roi_values)
    right = max(roi[0] + roi[2] for roi in roi_values)
    bottom = max(roi[1] + roi[3] for roi in roi_values)

    selected_width = max(1, right - left)
    selected_height = max(1, bottom - top)
    first_zone_height = max(1, roi_values[0][3])

    # Keep enough surrounding tube and equipment visible to preserve context,
    # while avoiding the extreme downscaling caused by showing the full frame.
    horizontal_padding = max(70, int(round(selected_width * 1.15)))
    vertical_padding = max(45, int(round(first_zone_height * 0.45)))

    crop_left = max(0, left - horizontal_padding)
    crop_top = max(0, top - vertical_padding)
    crop_right = min(image_width, right + horizontal_padding)
    crop_bottom = min(image_height, bottom + vertical_padding)

    # Avoid an excessively narrow preview on very tall videos.
    crop_width = max(1, crop_right - crop_left)
    crop_height = max(1, crop_bottom - crop_top)
    minimum_aspect = 0.32
    current_aspect = crop_width / crop_height
    if current_aspect < minimum_aspect:
        desired_width = int(round(crop_height * minimum_aspect))
        extra = max(0, desired_width - crop_width)
        crop_left = max(0, crop_left - extra // 2)
        crop_right = min(image_width, crop_right + extra - extra // 2)

    return image[crop_top:crop_bottom, crop_left:crop_right].copy()


def move_reference_top(delta: int, maximum_top: int) -> None:
    current = int(st.session_state.get("reference_top_input", 0))
    updated = min(max(current + delta, 0), maximum_top)
    st.session_state.reference_top_input = updated
    st.session_state.reference_top = updated
    clear_analysis_results()


def place_reference_below_main(default_top: int, maximum_top: int) -> None:
    updated = min(max(int(default_top), 0), maximum_top)
    st.session_state.reference_top_input = updated
    st.session_state.reference_top = updated
    clear_analysis_results()


# ==================================================
# ROI helpers
# ==================================================
def make_zone_label(index: int, zone_count: int) -> str:
    zone_number = index + 1
    if zone_count == 1:
        return "Zone 1"
    if index == 0:
        return "Zone 1 (Top)"
    if index == zone_count - 1:
        return f"Zone {zone_count} (Bottom)"
    return f"Zone {zone_number}"


def generate_zone_rois(main_box: dict, zone_count: int) -> Dict[str, tuple]:
    """Split one user-drawn boundary equally into the selected zones."""
    left = int(main_box["left"])
    top = int(main_box["top"])
    width = int(main_box["width"])
    height = int(main_box["height"])

    boundaries = np.rint(
        np.linspace(top, top + height, zone_count + 1)
    ).astype(int)

    rois = {}
    for index in range(zone_count):
        zone_top = int(boundaries[index])
        zone_bottom = int(boundaries[index + 1])
        zone_height = max(1, zone_bottom - zone_top)
        rois[make_zone_label(index, zone_count)] = (
            left,
            zone_top,
            width,
            zone_height,
        )

    return rois


def generate_reference_roi(reference_box: dict) -> Dict[str, tuple]:
    return {
        REFERENCE_ROI_LABEL: (
            int(reference_box["left"]),
            int(reference_box["top"]),
            int(reference_box["width"]),
            int(reference_box["height"]),
        )
    }


def roi_fits_inside_frame(
    roi: tuple,
    frame_width: int,
    frame_height: int,
) -> bool:
    x, y, width, height = roi
    return (
        x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= frame_width
        and y + height <= frame_height
    )


def rois_fit_inside_frame(
    rois: Dict[str, tuple],
    frame_width: int,
    frame_height: int,
) -> bool:
    return all(
        roi_fits_inside_frame(roi, frame_width, frame_height)
        for roi in rois.values()
    )


def get_display_label(name: str) -> str:
    return name.replace(" (Top)", "").replace(" (Bottom)", "")


def draw_readable_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    border_color: tuple,
    font_scale: float = 0.68,
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    text_thickness = 2
    padding_x = 7
    padding_y = 5

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness,
    )

    image_height, image_width = image.shape[:2]
    box_width = text_width + (2 * padding_x)
    box_height = text_height + baseline + (2 * padding_y)
    box_x = max(0, min(int(x), image_width - box_width - 1))
    box_y = max(0, min(int(y), image_height - box_height - 1))

    cv2.rectangle(
        image,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        image,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        border_color,
        2,
    )
    cv2.putText(
        image,
        text,
        (box_x + padding_x, box_y + padding_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_rois_on_frame(
    frame: np.ndarray,
    main_rois: Dict[str, tuple],
    reference_roi: Dict[str, tuple],
) -> np.ndarray:
    preview = frame.copy()
    main_color = (0, 255, 0)
    reference_color = (0, 255, 255)

    for name, (x, y, width, height) in main_rois.items():
        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            main_color,
            3,
        )
        draw_readable_label(
            preview,
            get_display_label(name),
            x + 4,
            y + 4,
            main_color,
        )

    for name, (x, y, width, height) in reference_roi.items():
        cv2.rectangle(
            preview,
            (x, y),
            (x + width, y + height),
            reference_color,
            3,
        )
        draw_readable_label(
            preview,
            name,
            x + 4,
            y + 4,
            reference_color,
        )

    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def build_roi_table(
    main_rois: Dict[str, tuple],
    reference_roi: Dict[str, tuple],
) -> pd.DataFrame:
    rows = []
    for name, (x, y, width, height) in main_rois.items():
        rows.append(
            {
                "ROI": name,
                "type": "main zone",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )

    for name, (x, y, width, height) in reference_roi.items():
        rows.append(
            {
                "ROI": name,
                "type": "reference",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )

    return pd.DataFrame(rows)


# ==================================================
# Analysis helpers
# ==================================================
def calculate_reference_values(
    first_frame: np.ndarray,
    rois: Dict[str, tuple],
) -> Dict[str, float]:
    # Keep the original float32 calculation for numerical consistency with
    # earlier versions of the dissertation prototype.
    gray_reference = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
    inverted_reference = 255 - gray_reference

    reference_values = {}
    for name, (x, y, width, height) in rois.items():
        roi_reference = inverted_reference[y:y + height, x:x + width]
        reference_values[name] = (
            float(np.mean(roi_reference))
            if roi_reference.size > 0
            else np.nan
        )

    return reference_values


def analyse_video(
    video_path: str,
    rois: Dict[str, tuple],
    rotation_angle: float,
    crop_box: dict,
    frame_step: int = 1,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("The video could not be opened.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError("The video FPS could not be read.")

    success, original_first_frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError("The first frame could not be read.")

    first_frame = preprocess_frame(
        original_first_frame,
        rotation_angle,
        crop_box,
    )
    output_height, output_width = first_frame.shape[:2]

    if not rois_fit_inside_frame(rois, output_width, output_height):
        cap.release()
        raise ValueError("At least one ROI is outside the prepared video frame.")

    reference_values = calculate_reference_values(first_frame, rois)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (output_width, output_height),
        isColor=True,
    )

    if not output_video.isOpened():
        cap.release()
        raise RuntimeError("The processed video could not be created.")

    density_results = {name: [] for name in rois}
    roi_items = list(rois.items())
    roi_draw_specs = [
        (
            name,
            roi,
            (0, 255, 255) if name == REFERENCE_ROI_LABEL else (0, 255, 0),
            get_display_label(name),
        )
        for name, roi in roi_items
    ]

    frame_index = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Browser/UI updates are relatively expensive on some Windows machines.
    # Limit them to roughly 100 updates for the whole video.
    progress_update_every = max(10, total_frames // 100) if total_frames > 0 else 25

    while True:
        success, original_frame = cap.read()
        if not success:
            break

        frame = preprocess_frame(
            original_frame,
            rotation_angle,
            crop_box,
        )

        if frame.shape[1] != output_width or frame.shape[0] != output_height:
            output_video.release()
            cap.release()
            raise RuntimeError("Prepared frame dimensions changed during analysis.")

        gray_frame_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted_frame_u8 = cv2.bitwise_not(gray_frame_u8)
        inverted_visualisation = cv2.cvtColor(
            inverted_frame_u8,
            cv2.COLOR_GRAY2BGR,
        )

        if frame_index % frame_step == 0:
            # Preserve the original float32 density calculation exactly. Only
            # create the float array on frames that are actually analysed.
            inverted_frame = inverted_frame_u8.astype(np.float32)
            time_seconds = frame_index / fps
            for name, (x, y, width, height) in roi_items:
                roi_frame = inverted_frame[y:y + height, x:x + width]
                density = (
                    float(np.mean(roi_frame))
                    if roi_frame.size > 0
                    else np.nan
                )
                density_results[name].append(
                    (frame_index, time_seconds, density)
                )

        for name, (x, y, width, height), box_color, display_label in roi_draw_specs:
            cv2.rectangle(
                inverted_visualisation,
                (x, y),
                (x + width, y + height),
                box_color,
                3,
            )
            draw_readable_label(
                inverted_visualisation,
                display_label,
                x + 4,
                y + 4,
                box_color,
                font_scale=0.72,
            )

        output_video.write(inverted_visualisation)
        frame_index += 1

        if total_frames > 0 and (
            frame_index == total_frames
            or frame_index % progress_update_every == 0
        ):
            progress_bar.progress(min(frame_index / total_frames, 1.0))
            status_text.text(
                f"Processing frame {frame_index} of {total_frames}"
            )

    cap.release()
    output_video.release()

    dataframes = []
    for roi_name, roi_data in density_results.items():
        roi_dataframe = pd.DataFrame(
            roi_data,
            columns=["frame", "time_sec", "density"],
        )
        roi_dataframe["ROI"] = roi_name
        roi_dataframe["reference_density"] = reference_values[roi_name]
        dataframes.append(roi_dataframe)

    if not dataframes:
        raise RuntimeError("No density data were generated.")

    complete_dataframe = pd.concat(dataframes, ignore_index=True)

    # Change is measured relative to the first-frame value of the same ROI.
    # Preserve the raw calculated change in the dataframe/CSV for traceability.
    # Negative change values are clipped to zero only when they are displayed
    # in the change-from-first-frame visualisation. Raw density is unchanged.
    complete_dataframe["change_from_first_frame"] = (
        complete_dataframe["density"]
        - complete_dataframe["reference_density"]
    )

    safe_baseline = complete_dataframe["reference_density"].replace(0, np.nan)
    complete_dataframe["percentage_change_from_first_frame"] = (
        complete_dataframe["change_from_first_frame"]
        / safe_baseline
        * 100.0
    )

    complete_dataframe = (
        complete_dataframe[
            [
                "frame",
                "time_sec",
                "ROI",
                "density",
                "reference_density",
                "change_from_first_frame",
                "percentage_change_from_first_frame",
            ]
        ]
        .sort_values(["ROI", "frame"])
        .reset_index(drop=True)
    )

    progress_bar.empty()
    status_text.text("Analysis completed.")

    return complete_dataframe, output_video_path, reference_values


# ==================================================
# Plot helpers
# ==================================================
def create_plot_dataframe(
    density_dataframe: pd.DataFrame,
    fps: float,
    frame_step: int,
    measurement_mode: str,
    smoothing_mode: str,
    smoothing_seconds: float,
) -> pd.DataFrame:
    plot_dataframe = density_dataframe.sort_values(["ROI", "frame"]).copy()

    if measurement_mode == "Raw density":
        plot_dataframe["chart_value"] = plot_dataframe["density"]
    else:
        # Orla's requested display rule: physically impossible negative change
        # values are shown as zero on the graph, while the CSV retains the
        # original calculated values for traceability.
        plot_dataframe["chart_value"] = plot_dataframe[
            "change_from_first_frame"
        ].clip(lower=0.0)

    if smoothing_mode == "Raw values":
        plot_dataframe["plot_value"] = plot_dataframe["chart_value"]
        return plot_dataframe

    effective_fps = fps / max(frame_step, 1)
    rolling_window = max(1, int(round(effective_fps * smoothing_seconds)))
    plot_dataframe["plot_value"] = (
        plot_dataframe.groupby("ROI")["chart_value"]
        .transform(
            lambda values: values.rolling(
                window=rolling_window,
                center=True,
            ).mean()
        )
    )
    return plot_dataframe


def choose_time_tick_intervals(time_values: pd.Series) -> Tuple[float, float]:
    """Choose readable major and minor time-axis intervals."""
    finite_values = pd.to_numeric(time_values, errors="coerce").dropna()

    if finite_values.empty:
        return 1.0, 0.2

    duration = float(finite_values.max() - finite_values.min())

    if duration <= 10:
        return 1.0, 0.2
    if duration <= 30:
        return 2.0, 0.5
    if duration <= 90:
        return 5.0, 1.0
    if duration <= 300:
        return 10.0, 2.0

    return 30.0, 5.0


def highlight_negative_value(value):
    """Highlight negative change values without colouring unrelated columns."""
    try:
        if pd.notna(value) and float(value) < 0:
            return "color: #991b1b; background-color: #fee2e2; font-weight: 600;"
    except (TypeError, ValueError):
        pass

    return ""


def build_styled_density_table(density_dataframe: pd.DataFrame):
    """Format the result table and highlight negative changes in red."""
    display_dataframe = density_dataframe.copy()

    formatter = {
        "frame": "{:.0f}",
        "time_sec": "{:.3f}",
        "density": "{:.4f}",
        "reference_density": "{:.4f}",
        "change_from_first_frame": "{:.4f}",
        "percentage_change_from_first_frame": "{:.4f}",
    }

    styled_table = display_dataframe.style.format(formatter, na_rep="—")
    negative_columns = [
        column
        for column in [
            "change_from_first_frame",
            "percentage_change_from_first_frame",
        ]
        if column in display_dataframe.columns
    ]

    if negative_columns:
        if hasattr(styled_table, "map"):
            styled_table = styled_table.map(
                highlight_negative_value,
                subset=negative_columns,
            )
        else:
            styled_table = styled_table.applymap(
                highlight_negative_value,
                subset=negative_columns,
            )

    return styled_table


def build_interactive_density_figure(
    plot_dataframe: pd.DataFrame,
    reference_values: Dict[str, float],
    selected_rois: list,
    measurement_mode: str,
    smoothing_mode: str,
    smoothing_seconds: float,
    show_reference_lines: bool,
    show_peak_markers: bool,
    show_overall_peak_guide: bool,
):
    """Build an interactive Plotly chart with hover values and peak guides."""
    figure = go.Figure()
    peak_rows = []
    palette = qualitative.Plotly

    if measurement_mode == "Raw density":
        y_axis_title = "Mean inverted brightness (density, a.u.)"
        hover_value_label = "Density"
        value_suffix = ""
        title = "Raw density vs time"
    else:
        y_axis_title = "Change from first frame (density, a.u.)"
        hover_value_label = "Change"
        value_suffix = ""
        title = "Change from first frame vs time"

    if smoothing_mode == "Rolling average":
        title += f" ({smoothing_seconds:.1f}s rolling average)"
    else:
        title += " (raw values)"

    for index, roi_name in enumerate(selected_rois):
        subset = (
            plot_dataframe[plot_dataframe["ROI"] == roi_name]
            .dropna(subset=["time_sec", "plot_value"])
            .sort_values("time_sec")
            .copy()
        )

        if subset.empty:
            continue

        line_colour = palette[index % len(palette)]

        figure.add_trace(
            go.Scatter(
                x=subset["time_sec"],
                y=subset["plot_value"],
                mode="lines",
                name=roi_name,
                line=dict(width=2.2, color=line_colour),
                hovertemplate=(
                    f"<b>{roi_name}</b><br>"
                    "Time: %{x:.3f} s<br>"
                    f"{hover_value_label}: %{{y:.4f}}{value_suffix}"
                    "<extra></extra>"
                ),
            )
        )

        peak_row = subset.loc[subset["plot_value"].idxmax()]
        peak_time = float(peak_row["time_sec"])
        peak_value = float(peak_row["plot_value"])

        peak_rows.append(
            {
                "ROI": roi_name,
                "Peak time (s)": peak_time,
                "Peak value": peak_value,
                "colour": line_colour,
            }
        )

        if show_peak_markers:
            figure.add_trace(
                go.Scatter(
                    x=[peak_time],
                    y=[peak_value],
                    mode="markers",
                    name=f"{roi_name} peak",
                    marker=dict(
                        size=10,
                        color=line_colour,
                        line=dict(width=1.5, color="white"),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{roi_name} peak</b><br>"
                        f"Time: {peak_time:.3f} s<br>"
                        f"{hover_value_label}: {peak_value:.4f}{value_suffix}"
                        "<extra></extra>"
                    ),
                )
            )

        if (
            measurement_mode == "Raw density"
            and show_reference_lines
            and roi_name in reference_values
            and np.isfinite(reference_values[roi_name])
        ):
            figure.add_hline(
                y=float(reference_values[roi_name]),
                line_dash="dash",
                line_width=1,
                line_color=line_colour,
                opacity=0.45,
            )

    if measurement_mode != "Raw density":
        figure.add_hline(
            y=0,
            line_dash="dash",
            line_width=1.4,
            line_color="black",
            opacity=0.8,
        )

    if peak_rows and show_overall_peak_guide:
        overall_peak = max(peak_rows, key=lambda row: row["Peak value"])
        figure.add_vline(
            x=overall_peak["Peak time (s)"],
            line_dash="dot",
            line_width=1.5,
            line_color="black",
            opacity=0.75,
            annotation_text=(
                f"Highest peak: {overall_peak['ROI']} "
                f"at {overall_peak['Peak time (s)']:.2f}s"
            ),
            annotation_position="top left",
        )

    major_tick, minor_tick = choose_time_tick_intervals(
        plot_dataframe.loc[
            plot_dataframe["ROI"].isin(selected_rois),
            "time_sec",
        ]
    )

    figure.update_layout(
        title=dict(
            text=title,
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        xaxis_title="Time (s)",
        yaxis_title=y_axis_title,
        template="plotly_white",
        # Show only the trace nearest to the pointer. The earlier
        # "x unified" mode listed every ROI at the same time point.
        hovermode="closest",
        hoverdistance=35,
        height=680,
        # A horizontal legend below the axis made Plotly reserve too much
        # vertical space when many zones were selected, collapsing the
        # actual plotting area. Keep a fixed vertical legend on the right.
        margin=dict(
            l=95,
            r=225,
            t=95,
            b=85,
            autoexpand=False,
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(120,120,120,0.25)",
            borderwidth=1,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            namelength=-1,
        ),
    )

    figure.update_xaxes(
        dtick=major_tick,
        title_standoff=18,
        automargin=False,
        ticks="outside",
        ticklen=7,
        showgrid=True,
        gridcolor="rgba(120, 120, 120, 0.28)",
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        minor=dict(
            dtick=minor_tick,
            ticks="inside",
            ticklen=4,
            showgrid=True,
            gridcolor="rgba(120, 120, 120, 0.11)",
            griddash="dot",
        ),
    )

    y_axis_updates = dict(
        title_standoff=18,
        automargin=False,
        ticks="outside",
        ticklen=7,
        showgrid=True,
        gridcolor="rgba(120, 120, 120, 0.28)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikethickness=1,
        minor=dict(
            ticks="inside",
            ticklen=4,
            showgrid=True,
            gridcolor="rgba(120, 120, 120, 0.11)",
            griddash="dot",
        ),
    )

    # Keep zero visible as the lower baseline for the change plot.
    if measurement_mode == "Change from first frame":
        y_axis_updates["rangemode"] = "tozero"

    figure.update_yaxes(**y_axis_updates)

    peak_dataframe = pd.DataFrame(peak_rows)
    if not peak_dataframe.empty:
        peak_dataframe = peak_dataframe.drop(columns=["colour"]).sort_values(
            "Peak value",
            ascending=False,
        )

    return figure, peak_dataframe


def build_wide_metric_table(
    density_dataframe: pd.DataFrame,
    value_column: str,
    include_reference: bool = True,
) -> pd.DataFrame:
    """Create one Excel-friendly time-series table with ROIs as headers."""
    base = density_dataframe.copy()
    if not include_reference:
        base = base[base["ROI"] != "Reference ROI"].copy()

    roi_order = list(dict.fromkeys(base["ROI"].tolist()))
    wide = base.pivot_table(
        index=["frame", "time_sec"],
        columns="ROI",
        values=value_column,
        aggfunc="first",
    ).reset_index()

    ordered_columns = ["frame", "time_sec"] + [
        roi for roi in roi_order if roi in wide.columns
    ]
    return wide[ordered_columns].sort_values(["frame", "time_sec"]).reset_index(drop=True)


def build_first_frame_baseline_table(density_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Store each ROI baseline once instead of repeating it at every time point."""
    baseline = (
        density_dataframe[["ROI", "reference_density"]]
        .drop_duplicates(subset=["ROI"], keep="first")
        .rename(columns={"reference_density": "first_frame_density"})
        .reset_index(drop=True)
    )
    return baseline


# ==================================================
# Sidebar and upload
# ==================================================
st.sidebar.header("Video Input")
uploaded_video = st.sidebar.file_uploader(
    "Upload a fluidisation video",
    type=["mp4", "mov", "avi", "mkv"],
)

st.sidebar.header("Analysis Settings")
frame_step = st.sidebar.number_input(
    "Frame step",
    min_value=1,
    max_value=100,
    value=1,
    help=(
        "1 analyses every frame. A larger value analyses fewer frames, "
        "but the processed output video still contains all frames."
    ),
)

if uploaded_video is None:
    st.info("Upload a video from the sidebar to begin.")
    st.stop()

uploaded_bytes = uploaded_video.getvalue()
current_video_id = hashlib.sha1(uploaded_bytes).hexdigest()[:16]

if st.session_state.uploaded_video_id != current_video_id:
    new_video_id, new_video_path = save_uploaded_video(uploaded_video)
    reset_for_new_video(
        video_id=new_video_id,
        filename=uploaded_video.name,
        video_path=new_video_path,
    )

video_path = st.session_state.video_path
video_info = get_video_info(video_path)
if video_info is None:
    st.error("The uploaded video could not be opened.")
    st.stop()

fps = video_info["fps"]
total_frames = video_info["total_frames"]
frame_width = video_info["frame_width"]
frame_height = video_info["frame_height"]


# ==================================================
# Compact video information
# ==================================================
st.caption(
    f"{fps:.2f} FPS  ·  {total_frames:,} frames  ·  "
    f"{frame_width} × {frame_height}  ·  {video_info['duration']:.2f} s"
)

# The setup preview frame is only for setup/positioning and does not change
# the first frame used by the analysis.
preview_frame_number = min(
    max(int(st.session_state.get("setup_preview_frame_number", 0)), 0),
    max(total_frames - 1, 0),
)
preview_size = (
    st.session_state.preview_size
    if st.session_state.preview_size in PREVIEW_MAX_HEIGHTS
    else "Medium"
)
preview_max_height = PREVIEW_MAX_HEIGHTS[preview_size]

original_preview_frame = get_frame(video_path, preview_frame_number)
if original_preview_frame is None:
    st.error("The selected preview frame could not be read.")
    st.stop()

# Always reconstruct the prepared frame from saved settings, even when a
# different wizard page is currently visible.
rotation_angle = float(st.session_state.rotation_angle)
crop_video_enabled = bool(st.session_state.crop_video_enabled)

rotated_preview_frame = rotate_frame_bound(
    original_preview_frame,
    rotation_angle,
)
rotated_height, rotated_width = rotated_preview_frame.shape[:2]

if st.session_state.video_crop_box is None:
    st.session_state.video_crop_box = default_crop_box(rotated_preview_frame)

video_crop_box = normalise_box(
    st.session_state.video_crop_box,
    rotated_width,
    rotated_height,
)
prepared_preview_frame = crop_frame(rotated_preview_frame, video_crop_box)
prepared_height, prepared_width = prepared_preview_frame.shape[:2]

if st.session_state.reference_roi_box is None:
    # Default reference is a small box near the lower centre of the prepared frame.
    ref_w = max(30, int(prepared_width * 0.22))
    ref_h = max(20, int(prepared_height * 0.07))
    st.session_state.reference_roi_box = {
        "left": max(0, int((prepared_width - ref_w) / 2)),
        "top": max(0, prepared_height - ref_h - max(5, int(prepared_height * 0.03))),
        "width": min(ref_w, prepared_width),
        "height": min(ref_h, prepared_height),
    }

if st.session_state.main_area_box is None:
    ref_box = normalise_box(
        st.session_state.reference_roi_box,
        prepared_width,
        prepared_height,
    )
    main_top = max(0, int(prepared_height * 0.08))
    main_bottom = max(main_top + 20, ref_box["top"] - 4)
    st.session_state.main_area_box = {
        "left": ref_box["left"],
        "top": main_top,
        "width": ref_box["width"],
        "height": max(20, main_bottom - main_top),
    }

workflow_step = int(st.session_state.get("workflow_step", 1))
workflow_step = min(max(workflow_step, 1), 4)
st.session_state.workflow_step = workflow_step

# ==================================================
# STEP 1 — Prepare video
# ==================================================
if workflow_step == 1:
    st.subheader("Step 1 — Prepare Video")

    setup_left, setup_right = st.columns([0.34, 0.66], gap="large")

    with setup_left:
        st.markdown("#### Prepare the video")
        st.write(
            "Choose a clear frame, align the tube vertically, then crop the useful area. "
            "Keep both the tube and the stationary reference area visible."
        )
        st.caption("The Reference ROI will be selected in Step 2.")

        preview_frame_input = st.slider(
            "Setup preview frame",
            min_value=0,
            max_value=max(total_frames - 1, 0),
            value=preview_frame_number,
            step=1,
            help=(
                "Choose a clear frame for rotation, crop and ROI setup. "
                "This does not change the first frame used by the analysis."
            ),
        )
        if int(preview_frame_input) != int(st.session_state.setup_preview_frame_number):
            st.session_state.setup_preview_frame_number = int(preview_frame_input)
            st.rerun()

        preview_time = (float(preview_frame_input) / fps) if fps > 0 else 0.0
        st.caption(
            f"Frame {int(preview_frame_input):,} of {max(total_frames - 1, 0):,} "
            f"· {preview_time:.2f} s · setup preview only"
        )

        preview_size_input = st.selectbox(
            "Preview size",
            options=list(PREVIEW_MAX_HEIGHTS.keys()),
            index=list(PREVIEW_MAX_HEIGHTS.keys()).index(preview_size),
            format_func=lambda option: PREVIEW_SIZE_LABELS[option],
            help=(
                "This changes only the on-screen preview size, not the video "
                "resolution or analysis accuracy."
            ),
        )
        if preview_size_input != st.session_state.preview_size:
            st.session_state.preview_size = preview_size_input
            st.rerun()

        rotation_angle_input = st.number_input(
            "Rotation angle (degrees)",
            min_value=-15.0,
            max_value=15.0,
            value=float(st.session_state.rotation_angle),
            step=0.2,
            help="Positive values rotate anticlockwise.",
        )
        crop_enabled_input = st.checkbox(
            "Crop video before ROI selection",
            value=bool(st.session_state.crop_video_enabled),
        )

        if (
            float(rotation_angle_input) != float(st.session_state.rotation_angle)
            or bool(crop_enabled_input) != bool(st.session_state.crop_video_enabled)
        ):
            st.session_state.rotation_angle = float(rotation_angle_input)
            st.session_state.crop_video_enabled = bool(crop_enabled_input)
            st.session_state.video_crop_box = None
            st.session_state.video_crop_pending = None
            st.session_state.crop_applied = False
            st.session_state.reference_roi_box = None
            st.session_state.reference_roi_pending = None
            st.session_state.reference_roi_applied = False
            st.session_state.main_area_box = None
            st.session_state.main_area_pending = None
            st.session_state.main_area_applied = False
            st.session_state.processed_signature = None
            clear_analysis_results()
            st.rerun()

    preview_frame_number = int(st.session_state.setup_preview_frame_number)
    original_preview_frame = get_frame(video_path, preview_frame_number)
    if original_preview_frame is None:
        st.error("The selected setup preview frame could not be read.")
        st.stop()

    rotation_angle = float(st.session_state.rotation_angle)
    crop_video_enabled = bool(st.session_state.crop_video_enabled)
    preview_size = st.session_state.preview_size
    preview_max_height = PREVIEW_MAX_HEIGHTS[preview_size]
    rotated_preview_frame = rotate_frame_bound(original_preview_frame, rotation_angle)
    rotated_height, rotated_width = rotated_preview_frame.shape[:2]

    if st.session_state.video_crop_box is None:
        st.session_state.video_crop_box = default_crop_box(rotated_preview_frame)

    # IMPORTANT: pass the full-resolution working frame directly to
    # streamlit-cropper. The component may resize it for display internally,
    # but when should_resize_image=True it converts the returned box back to
    # coordinates of this input image. This keeps one coordinate system across
    # machines and avoids browser/display-width dependent scaling.
    crop_display_source = add_vertical_alignment_guide(rotated_preview_frame)

    with setup_right:
        st.markdown("#### Video workspace")
        if crop_video_enabled:
            st.caption(
                "Drag the blue box around the useful area, then click "
                "**Apply Crop & Continue**."
            )

            applied_crop_box = normalise_box(
                st.session_state.video_crop_box,
                rotated_width,
                rotated_height,
            )

            # Resize once to a known canvas and place the custom component in a
            # Streamlit form. The browser can now move/resize the box freely
            # without rerunning Python on every mouse movement. Coordinates are
            # submitted exactly once when an Apply button is pressed.
            crop_canvas, crop_scale_x, crop_scale_y = resize_frame_for_display(
                crop_display_source,
                max_height=700,
                max_width=480,
            )
            crop_default_display = scale_box_to_display(
                applied_crop_box,
                crop_scale_x,
                crop_scale_y,
            )

            crop_form_key = (
                f"crop_form_{current_video_id}_{rotation_angle:.1f}_"
                f"rev{int(st.session_state.crop_revision)}"
            )
            with st.form(crop_form_key, clear_on_submit=False):
                returned_crop_display = st_cropper(
                    bgr_to_pil(crop_canvas),
                    realtime_update=True,
                    default_coords=box_to_default_coords(crop_default_display),
                    box_color="#2E86DE",
                    aspect_ratio=None,
                    return_type="box",
                    key=f"{crop_form_key}_cropper",
                    should_resize_image=False,
                    stroke_width=4,
                )
                crop_continue_clicked = st.form_submit_button(
                    "Apply Crop & Continue →",
                    type="primary",
                    use_container_width=True,
                )

            if crop_continue_clicked:
                submitted_crop_box = normalise_box(
                    scale_box_to_original(
                        returned_crop_display,
                        crop_scale_x,
                        crop_scale_y,
                    ),
                    rotated_width,
                    rotated_height,
                )
                st.session_state.video_crop_box = submitted_crop_box.copy()
                st.session_state.video_crop_pending = submitted_crop_box.copy()
                st.session_state.crop_applied = True
                st.session_state.crop_revision = int(st.session_state.crop_revision) + 1

                # The coordinate system changes after a new crop, so downstream
                # ROIs must be selected on the newly cropped frame.
                st.session_state.reference_roi_box = None
                st.session_state.reference_roi_pending = None
                st.session_state.reference_roi_applied = False
                st.session_state.main_area_box = None
                st.session_state.main_area_pending = None
                st.session_state.main_area_applied = False
                st.session_state.processed_signature = None
                clear_analysis_results()

                st.session_state.workflow_step = 2
                st.rerun()

            crop_is_dirty = not bool(st.session_state.crop_applied)
            if bool(st.session_state.crop_applied):
                applied_crop_box = normalise_box(
                    st.session_state.video_crop_box,
                    rotated_width,
                    rotated_height,
                )
                st.success(
                    f"Crop applied: {applied_crop_box['width']} × "
                    f"{applied_crop_box['height']} px."
                )
                st.caption(
                    "Move the blue box if needed, then use **Apply Crop & Continue** "
                    "to confirm the latest position."
                )
            else:
                st.warning(
                    "Crop not applied yet. Move the blue box, then click **Apply Crop & Continue**."
                )

        else:
            pending_crop_box = full_frame_box(rotated_preview_frame)
            st.session_state.video_crop_box = pending_crop_box.copy()
            st.session_state.video_crop_pending = pending_crop_box.copy()
            st.session_state.crop_applied = True
            crop_is_dirty = False

            full_display_frame, _, _ = resize_frame_for_display(
                crop_display_source,
                max_height=max(preview_max_height, 650),
                max_width=700,
            )
            st.image(
                bgr_to_pil(full_display_frame),
                caption="The full rotated frame will be used.",
            )

    if not crop_video_enabled:
        nav_left, nav_right = st.columns([1, 1])
        with nav_right:
            if st.button(
                "Next → Select Reference ROI",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.workflow_step = 2
                st.rerun()


# ==================================================
# STEP 2 — Select reference ROI first
# ==================================================
elif workflow_step == 2:
    st.subheader("Step 2 — Select the Reference ROI")

    # IMPORTANT: Step 2 is reconstructed from the APPLIED Step 1 crop only.
    # Crop coordinates are submitted only when an Apply button is clicked.
    prepared_preview_frame = crop_frame(
        rotated_preview_frame,
        normalise_box(
            st.session_state.video_crop_box,
            rotated_width,
            rotated_height,
        ),
    )
    prepared_height, prepared_width = prepared_preview_frame.shape[:2]

    if st.session_state.reference_roi_box is None:
        ref_w = max(30, int(prepared_width * 0.22))
        ref_h = max(20, int(prepared_height * 0.07))
        st.session_state.reference_roi_box = {
            "left": max(0, int((prepared_width - ref_w) / 2)),
            "top": max(0, prepared_height - ref_h - max(5, int(prepared_height * 0.03))),
            "width": min(ref_w, prepared_width),
            "height": min(ref_h, prepared_height),
        }

    applied_ref_box = normalise_box(
        st.session_state.reference_roi_box,
        prepared_width,
        prepared_height,
    )
    pending_ref_source = (
        st.session_state.reference_roi_pending
        if st.session_state.reference_roi_pending is not None
        else applied_ref_box
    )
    pending_ref_source = normalise_box(
        pending_ref_source,
        prepared_width,
        prepared_height,
    )

    ref_help_col, ref_workspace_col = st.columns([0.32, 0.68], gap="large")
    with ref_help_col:
        st.markdown("#### How to choose the Reference ROI")
        st.write(
            "Draw the **yellow box** around a stationary area at the bottom of the tube."
        )
        st.markdown(
            "- Choose an area that remains unchanged.\n"
            "- Keep it outside the moving particle region.\n"
            "- Place it close to where the analysis region will begin."
        )
        st.caption("Yellow = Reference ROI")

    with ref_workspace_col:
        ref_spacer_col, ref_cropper_col = st.columns([0.18, 0.82], gap="small")
        with ref_cropper_col:
            st.markdown("#### Reference workspace")
            st.caption(
                "Move/resize the yellow box, then click **Apply Reference ROI & Continue**."
            )
            ref_canvas, ref_scale_x, ref_scale_y = resize_frame_for_display(
                prepared_preview_frame,
                max_height=700,
                max_width=480,
            )
            ref_default_display = scale_box_to_display(
                applied_ref_box,
                ref_scale_x,
                ref_scale_y,
            )
            ref_form_key = (
                f"ref_form_{current_video_id}_"
                f"crop{int(st.session_state.crop_revision)}_"
                f"rev{int(st.session_state.reference_revision)}"
            )
            with st.form(ref_form_key, clear_on_submit=False):
                returned_ref_display = st_cropper(
                    bgr_to_pil(ref_canvas),
                    realtime_update=True,
                    default_coords=box_to_default_coords(ref_default_display),
                    box_color="#FFD54F",
                    aspect_ratio=None,
                    return_type="box",
                    key=f"{ref_form_key}_cropper",
                    should_resize_image=False,
                    stroke_width=4,
                )
                ref_continue_clicked = st.form_submit_button(
                    "Apply Reference ROI & Continue →",
                    type="primary",
                    use_container_width=True,
                )

            if ref_continue_clicked:
                submitted_ref_box = normalise_box(
                    scale_box_to_original(
                        returned_ref_display,
                        ref_scale_x,
                        ref_scale_y,
                    ),
                    prepared_width,
                    prepared_height,
                )
                st.session_state.reference_roi_box = submitted_ref_box.copy()
                st.session_state.reference_roi_pending = submitted_ref_box.copy()
                st.session_state.reference_roi_applied = True
                st.session_state.reference_revision = int(st.session_state.reference_revision) + 1
                st.session_state.main_area_box = None
                st.session_state.main_area_pending = None
                st.session_state.main_area_applied = False
                st.session_state.processed_signature = None
                clear_analysis_results()

                st.session_state.workflow_step = 3
                st.rerun()

    if bool(st.session_state.reference_roi_applied):
        st.success("Reference ROI applied.")
        st.caption(
            "Move the yellow box if needed, then use **Apply Reference ROI & Continue** "
            "to confirm the latest position."
        )
    else:
        st.warning(
            "Reference ROI not applied yet. Move the yellow box, then click "
            "**Apply Reference ROI & Continue**."
        )

    back_col, _ = st.columns(2)
    with back_col:
        if st.button("← Back to Video Preparation", use_container_width=True):
            st.session_state.workflow_step = 1
            st.rerun()


# ==================================================
# STEP 3 — Select main analysis area above reference
# ==================================================
elif workflow_step == 3:
    st.subheader("Step 3 — Select the Analysis Area and Zones")

    # Rebuild the prepared frame from the APPLIED crop and use the APPLIED
    # Reference ROI. Green-box drag events are batched until Apply is clicked.
    prepared_preview_frame = crop_frame(
        rotated_preview_frame,
        normalise_box(
            st.session_state.video_crop_box,
            rotated_width,
            rotated_height,
        ),
    )
    prepared_height, prepared_width = prepared_preview_frame.shape[:2]

    reference_roi_box = normalise_box(
        st.session_state.reference_roi_box,
        prepared_width,
        prepared_height,
    )
    reference_roi = generate_reference_roi(reference_roi_box)

    main_base_frame = prepared_preview_frame.copy()
    ref_left = int(reference_roi_box["left"])
    ref_top = int(reference_roi_box["top"])
    ref_right = int(reference_roi_box["left"] + reference_roi_box["width"])
    ref_bottom = int(reference_roi_box["top"] + reference_roi_box["height"])
    cv2.rectangle(
        main_base_frame,
        (ref_left, ref_top),
        (ref_right, ref_bottom),
        (0, 215, 255),
        5,
    )
    cv2.putText(
        main_base_frame,
        "Reference ROI",
        (ref_left, max(28, ref_top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    if st.session_state.main_area_box is None:
        main_top = max(0, int(prepared_height * 0.08))
        main_bottom = max(main_top + 20, reference_roi_box["top"] - 4)
        st.session_state.main_area_box = {
            "left": reference_roi_box["left"],
            "top": main_top,
            "width": reference_roi_box["width"],
            "height": max(20, main_bottom - main_top),
        }

    applied_main_box = normalise_box(
        st.session_state.main_area_box,
        prepared_width,
        prepared_height,
        minimum_size=2,
    )

    main_canvas, main_scale_x, main_scale_y = resize_frame_for_display(
        main_base_frame,
        max_height=700,
        max_width=480,
    )
    main_default_display = scale_box_to_display(
        applied_main_box,
        main_scale_x,
        main_scale_y,
    )
    main_form_key = (
        f"main_form_{current_video_id}_"
        f"crop{int(st.session_state.crop_revision)}_"
        f"ref{int(st.session_state.reference_revision)}"
    )

    # Keep both the zone setting and green cropper in ONE form. Changing the
    # zone count therefore does not rerun the page or reset an unsaved green
    # box position. Both values are committed together by one button.
    with st.form(main_form_key, clear_on_submit=False):
        zone_help_col, zone_workspace_col = st.columns([0.32, 0.68], gap="large")

        with zone_help_col:
            st.markdown("#### Analysis settings")
            st.write(
                "Draw the **green box** around the part of the tube you want to analyse. "
                "Keep it directly above the yellow Reference ROI."
            )
            st.caption("Yellow = Reference ROI · Green = Analysis area")

            zone_count = st.number_input(
                "Number of zones",
                min_value=1,
                max_value=100,
                value=min(max(int(st.session_state.zone_count), 1), 100),
                step=1,
                help=(
                    "The green analysis area is divided into equal horizontal zones. "
                    "More zones provide finer spatial resolution but generate more data."
                ),
            )
            st.caption(
                "Changing this value will not move the green box. The zone count and "
                "green area are confirmed together when you continue."
            )

        with zone_workspace_col:
            zone_spacer_col, zone_cropper_col = st.columns([0.18, 0.82], gap="small")
            with zone_cropper_col:
                st.markdown("#### Analysis workspace")
                st.caption(
                    "Move/resize the green box and choose the number of zones, then "
                    "click **Apply Analysis Area & Continue**."
                )
                returned_main_display = st_cropper(
                    bgr_to_pil(main_canvas),
                    realtime_update=True,
                    default_coords=box_to_default_coords(main_default_display),
                    box_color="#00C853",
                    aspect_ratio=None,
                    return_type="box",
                    key=f"{main_form_key}_cropper",
                    should_resize_image=False,
                    stroke_width=4,
                )
                main_continue_clicked = st.form_submit_button(
                    "Apply Analysis Area & Continue →",
                    type="primary",
                    use_container_width=True,
                )

    if main_continue_clicked:
        submitted_main_box = normalise_box(
            scale_box_to_original(
                returned_main_display,
                main_scale_x,
                main_scale_y,
            ),
            prepared_width,
            prepared_height,
            minimum_size=2,
        )
        submitted_zone_count = int(zone_count)

        st.session_state.main_area_box = submitted_main_box.copy()
        st.session_state.main_area_pending = submitted_main_box.copy()
        st.session_state.main_area_applied = True
        st.session_state.zone_count = submitted_zone_count
        st.session_state.processed_signature = None
        clear_analysis_results()

        submitted_bottom = submitted_main_box["top"] + submitted_main_box["height"]
        submitted_overlap = not (
            submitted_bottom <= reference_roi_box["top"]
            or submitted_main_box["top"] >= reference_roi_box["top"] + reference_roi_box["height"]
            or submitted_main_box["left"] + submitted_main_box["width"] <= reference_roi_box["left"]
            or submitted_main_box["left"] >= reference_roi_box["left"] + reference_roi_box["width"]
        )
        submitted_zone_count_too_high = submitted_zone_count > int(submitted_main_box["height"])

        if not submitted_overlap and not submitted_zone_count_too_high:
            st.session_state.workflow_step = 4
        st.rerun()

    # The form batches drag events, so Python only sees a new position after
    # Apply. Validation therefore uses the latest applied box.
    main_area_box_for_validation = normalise_box(
        st.session_state.main_area_box,
        prepared_width,
        prepared_height,
        minimum_size=2,
    )
    main_bottom = (
        main_area_box_for_validation["top"]
        + main_area_box_for_validation["height"]
    )
    ref_top = reference_roi_box["top"]
    overlap = not (
        main_bottom <= reference_roi_box["top"]
        or main_area_box_for_validation["top"] >= reference_roi_box["top"] + reference_roi_box["height"]
        or main_area_box_for_validation["left"] + main_area_box_for_validation["width"] <= reference_roi_box["left"]
        or main_area_box_for_validation["left"] >= reference_roi_box["left"] + reference_roi_box["width"]
    )

    zone_count_too_high = int(zone_count) > int(main_area_box_for_validation["height"])
    if zone_count_too_high:
        st.error(
            f"The selected analysis area is {main_area_box_for_validation['height']} px high, "
            f"so it cannot be split reliably into {int(zone_count)} zones. "
            "Use fewer zones or make the green area taller."
        )

    if overlap:
        st.error(
            "The green analysis area overlaps the yellow Reference ROI. "
            "Move the green box above the reference so the two regions do not overlap."
        )
    elif main_bottom > ref_top:
        st.warning(
            "The analysis area extends below the top of the Reference ROI. "
            "Keep the green analysis area above the yellow reference."
        )
    else:
        gap_px = ref_top - main_bottom
        if gap_px <= 8:
            st.success(
                "Good alignment: the green analysis area starts directly above the yellow Reference ROI."
            )
        else:
            st.info(
                f"The analysis area is {gap_px} px above the Reference ROI. "
                "You can move the green box closer if you want to analyse particles "
                "immediately above the stationary region."
            )

    if bool(st.session_state.main_area_applied):
        st.success("Analysis area applied.")
        st.caption(
            "Changing **Number of zones** does not move the green area. It only "
            "changes how this applied area is divided."
        )
        if overlap or zone_count_too_high:
            st.warning(
                "Adjust the green area or zone count, then click **Apply Analysis Area & Continue**."
            )
    else:
        st.warning(
            "Analysis area not applied yet. Move the green box, choose the zone count, "
            "then click **Apply Analysis Area & Continue**."
        )

    back_col, _ = st.columns(2)
    with back_col:
        if st.button("← Back to Reference ROI", use_container_width=True):
            st.session_state.workflow_step = 2
            st.rerun()


# ==================================================
# Shared ROI reconstruction for analysis/results page
# ==================================================
reference_roi_box = normalise_box(
    st.session_state.reference_roi_box,
    prepared_width,
    prepared_height,
)
main_area_box = normalise_box(
    st.session_state.main_area_box,
    prepared_width,
    prepared_height,
    minimum_size=2,
)
zone_count = int(st.session_state.zone_count)
main_rois = generate_zone_rois(main_area_box, zone_count)
reference_roi = generate_reference_roi(reference_roi_box)
all_rois = {**main_rois, **reference_roi}
all_rois_are_valid = rois_fit_inside_frame(
    all_rois,
    prepared_width,
    prepared_height,
)

roi_table = build_roi_table(main_rois, reference_roi)
preparation_table = pd.DataFrame(
    [
        {
            "rotation_angle_degrees": float(st.session_state.rotation_angle),
            "crop_left": video_crop_box["left"],
            "crop_top": video_crop_box["top"],
            "crop_width": video_crop_box["width"],
            "crop_height": video_crop_box["height"],
            "number_of_zones": zone_count,
            "reference_roi_included": True,
            "reference_left": reference_roi_box["left"],
            "reference_top": reference_roi_box["top"],
            "reference_width": reference_roi_box["width"],
            "reference_height": reference_roi_box["height"],
        }
    ]
)

# Keep the analysis variables aligned with the saved preparation settings.
rotation_angle = float(st.session_state.rotation_angle)

# ==================================================
# STEP 4 — Run analysis
# ==================================================
if workflow_step == 4:
    st.subheader("Step 4 — Run Analysis")
    st.write(
        "The analysis uses the prepared video, the manually selected Reference ROI "
        "and the analysis zones shown below."
    )

    final_preview = draw_roi_setup_preview(
        prepared_preview_frame,
        main_rois,
        reference_roi,
    )
    final_preview_bgr = cv2.cvtColor(final_preview, cv2.COLOR_RGB2BGR)
    final_display, _, _ = resize_frame_for_display(
        final_preview_bgr,
        max_height=preview_max_height,
        max_width=820,
    )
    st.image(
        cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB),
        caption=f"Final setup: {zone_count} green analysis zones + yellow Reference ROI.",
        width=final_display.shape[1],
    )

    nav_back, nav_run = st.columns(2)
    with nav_back:
        if st.button("← Back to Analysis Area", use_container_width=True):
            st.session_state.workflow_step = 3
            st.rerun()

    with nav_run:
        run_analysis = st.button(
            "Run Analysis",
            type="primary",
            disabled=not all_rois_are_valid,
            use_container_width=True,
        )
else:
    run_analysis = False

if run_analysis:
    try:
        density_dataframe, processed_video_path, reference_values = analyse_video(
            video_path=video_path,
            rois=all_rois.copy(),
            rotation_angle=float(rotation_angle),
            crop_box=video_crop_box.copy(),
            frame_step=int(frame_step),
        )

        st.session_state.analysis_results = {
            "density_dataframe": density_dataframe,
            "processed_video_path": processed_video_path,
            "reference_values": reference_values,
            "fps": fps,
            "frame_step": int(frame_step),
            "roi_table": roi_table.copy(),
            "preparation_table": preparation_table.copy(),
            "main_roi_labels": list(main_rois.keys()),
            "all_roi_labels": list(all_rois.keys()),
        }
        st.success("Analysis completed.")

    except Exception as error:
        st.session_state.analysis_results = None
        st.error(f"Analysis failed: {error}")


# ==================================================
# 5. Results
# ==================================================
if workflow_step == 4 and st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    density_dataframe = results["density_dataframe"]
    processed_video_path = results["processed_video_path"]
    reference_values = results["reference_values"]
    result_fps = results["fps"]
    result_frame_step = results["frame_step"]
    result_roi_table = results["roi_table"]
    result_preparation_table = results["preparation_table"]
    main_roi_labels = results["main_roi_labels"]
    all_roi_labels = results["all_roi_labels"]

    st.subheader("5. Results")
    st.caption(
        "For change-based outputs, values below the first-frame baseline are shown as 0. "
        "Raw density values remain unchanged."
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Analysed ROIs",
        density_dataframe["ROI"].nunique(),
    )
    summary_columns[1].metric("Analysed samples", len(density_dataframe))
    summary_columns[2].metric("Frame step", result_frame_step)
    summary_columns[3].metric(
        "Mean density",
        f"{density_dataframe['density'].mean():.2f}",
    )

    with st.expander("First-frame baseline density values"):
        reference_dataframe = pd.DataFrame(
            [
                {"ROI": name, "reference_density": value}
                for name, value in reference_values.items()
            ]
        )
        st.dataframe(
            reference_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Density time-series data"):
        table_preview = density_dataframe.head(1000)
        st.dataframe(
            table_preview,
            use_container_width=True,
            hide_index=True,
            height=460,
        )
        st.caption(
            f"Showing the first {len(table_preview):,} of {len(density_dataframe):,} rows. "
            "The CSV download contains the complete dataset."
        )

    st.markdown("### Density chart")
    measurement_mode = st.radio(
        "Chart measure",
        options=[
            "Change from first frame",
            "Raw density",
        ],
        index=0,
        horizontal=True,
        help=(
            "Change from first frame uses each ROI's own first-frame density as "
            "its baseline. Negative calculated changes are displayed as zero."
        ),
    )

    smoothing_mode = st.radio(
        "Line display",
        options=["Raw values", "Rolling average"],
        index=0,
        horizontal=True,
    )

    if smoothing_mode == "Rolling average":
        smoothing_seconds = st.slider(
            "Rolling-average window (seconds)",
            min_value=0.1,
            max_value=5.0,
            value=2.0,
            step=0.1,
        )
    else:
        smoothing_seconds = 0.0

    selected_rois = st.multiselect(
        "ROIs shown in the chart",
        options=all_roi_labels,
        default=main_roi_labels,
    )

    if measurement_mode == "Raw density":
        show_reference_lines = st.checkbox(
            "Show first-frame reference lines",
            value=True,
        )
    else:
        show_reference_lines = False
        st.info(
            "Each displayed ROI uses its own first-frame value as the baseline. "
            "The dashed zero line is the shared baseline."
        )

    chart_option_columns = st.columns(2)
    with chart_option_columns[0]:
        show_peak_markers = st.checkbox(
            "Show peak markers",
            value=True,
            help=(
                "Mark the highest displayed value for each selected ROI. "
                "Peak detection uses the currently displayed raw or smoothed curve."
            ),
        )
    with chart_option_columns[1]:
        show_overall_peak_guide = st.checkbox(
            "Show vertical guide at the highest peak",
            value=True,
            help=(
                "Draw one dotted vertical line at the highest peak among the "
                "currently selected ROI curves."
            ),
        )

    if not selected_rois:
        st.warning("Select at least one ROI to display the chart.")
    else:
        plot_dataframe = create_plot_dataframe(
            density_dataframe=density_dataframe,
            fps=result_fps,
            frame_step=result_frame_step,
            measurement_mode=measurement_mode,
            smoothing_mode=smoothing_mode,
            smoothing_seconds=smoothing_seconds,
        )

        density_figure, peak_dataframe = build_interactive_density_figure(
            plot_dataframe=plot_dataframe,
            reference_values=reference_values,
            selected_rois=selected_rois,
            measurement_mode=measurement_mode,
            smoothing_mode=smoothing_mode,
            smoothing_seconds=smoothing_seconds,
            show_reference_lines=show_reference_lines,
            show_peak_markers=show_peak_markers,
            show_overall_peak_guide=show_overall_peak_guide,
        )

        st.plotly_chart(
            density_figure,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        st.caption(
            "Move the pointer onto a curve to see that ROI only. "
            "Drag to zoom, double-click to reset, and click legend items to "
            "hide or show individual ROI curves."
        )

        if not peak_dataframe.empty:
            with st.expander("Peak summary"):
                peak_display = peak_dataframe.copy()
                peak_display["Peak time (s)"] = peak_display[
                    "Peak time (s)"
                ].round(3)
                peak_display["Peak value"] = peak_display["Peak value"].round(4)
                st.dataframe(
                    peak_display,
                    use_container_width=True,
                    hide_index=True,
                )

        if smoothing_mode == "Rolling average":
            st.caption(
                "The centred rolling average may leave the beginning and end "
                "of each curve blank. Peak markers are calculated from the "
                "displayed rolling-average curve."
            )

    st.subheader("Downloads")
    st.caption(
        "Negative change values are retained in the Change CSV as raw calculated "
        "values for traceability, while the change graph displays negative values as zero."
    )

    raw_density_wide = build_wide_metric_table(
        density_dataframe,
        "density",
        include_reference=True,
    )
    raw_density_csv = raw_density_wide.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download raw density CSV",
        data=raw_density_csv,
        file_name="raw_density_zones_as_columns.csv",
        mime="text/csv",
        help=(
            "One time point per row. Reference ROI and analysis zones are shown "
            "once as column headers."
        ),
    )

    change_wide = build_wide_metric_table(
        density_dataframe,
        "change_from_first_frame",
        include_reference=True,
    )
    change_csv = change_wide.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download change-from-first-frame CSV",
        data=change_csv,
        file_name="change_from_first_frame_zones_as_columns.csv",
        mime="text/csv",
        help=(
            "Raw calculated change values are retained here, including negative "
            "values. The visualisation clips negative change values to zero."
        ),
    )

    baseline_table = build_first_frame_baseline_table(density_dataframe)
    baseline_csv = baseline_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download first-frame baseline CSV",
        data=baseline_csv,
        file_name="first_frame_baseline_by_roi.csv",
        mime="text/csv",
        help=(
            "One row per ROI. Each first-frame density is stored once rather than "
            "being repeated for every frame."
        ),
    )

    roi_settings_csv = result_roi_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ROI settings",
        data=roi_settings_csv,
        file_name="roi_settings.csv",
        mime="text/csv",
    )

    preparation_csv = result_preparation_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download video preparation settings",
        data=preparation_csv,
        file_name="video_preparation_settings.csv",
        mime="text/csv",
    )

    if os.path.exists(processed_video_path):
        with open(processed_video_path, "rb") as processed_video_file:
            processed_video_bytes = processed_video_file.read()

        st.download_button(
            label="Download prepared and processed ROI video",
            data=processed_video_bytes,
            file_name="prepared_inverted_with_ROIs.mp4",
            mime="video/mp4",
        )

        st.write("Prepared inverted video with ROI boxes")
        st.video(processed_video_bytes)
    else:
        st.warning("The temporary processed video file is no longer available.")
