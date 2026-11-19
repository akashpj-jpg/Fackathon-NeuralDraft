"""
Engineering Graphics Auto-Grader — Cloud-Ready Portal (DXF + OpenCV + Gemini VLM + Streamlit)
=============================================================================================
Single self-contained Streamlit web application with:
  1. Student Submission Tab: Name, U_ID, Email, Scan Upload -> automated grading + instant feedback
  2. Password-Protected Teacher Dashboard Tab: tabulated rubric gradebook, CSV & Excel (.xlsx)
     export, side-by-side overlay inspection, faculty mark overrides, and grading/DXF settings
  3. Borderless DXF + OpenCV engine + optional Gemini VLM notation & Isometric Scale audit
  4. Persistent SQLite gradebook (works out-of-the-box) + optional Google Sheets cloud sync

Local Install (inside .venv):
    pip install -r requirements.txt

Launch Web App:
    streamlit run eg_autograder_master.py
"""

import argparse
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import ezdxf
except ImportError:
    ezdxf = None

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ---------------------------------------------------------------------------
# Constants & Storage Paths
# ---------------------------------------------------------------------------
ISO_FACTOR = math.sqrt(2.0 / 3.0)       # 0.81649658 = cos(45)/cos(30) per BIS SP 46:2003
VIEW_OVER_PROJ = 1.0 / ISO_FACTOR       # 1.22474487 (maps v3 DXF to 1:1 Isometric View)

RUBRIC_WEIGHTS = {
    "scale_projection_vs_view": 2.0,
    "visible_cube": 3.0,
    "visible_prism": 3.0,
    "hidden_edges": 1.0,
    "dimensions": 1.0,
}

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "eg_submissions.db"
ACTIVE_DXF_PATH = BASE_DIR / "active_reference.dxf"

DEFAULT_PROBLEM_STATEMENT = (
    "A square prism of base side 40 mm and height 50 mm is centrally placed on top of a cube "
    "of side 60 mm. Draw the isometric projection of the solid."
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "problem_statement": DEFAULT_PROBLEM_STATEMENT,
    "calib_mode": "Page Height (mm)",
    "calib_val": 290.0,
    "tol_mm": 2.5,
    "hidden_policy": "omit",
    "enable_vlm": True,
    "model_id": "gemini-2.5-flash",
    "show_results_to_students": True,
}

Segment = Tuple[Tuple[float, float], Tuple[float, float]]


# ---------------------------------------------------------------------------
# Secrets & Environment Helpers (Streamlit Cloud + Local Compatible)
# ---------------------------------------------------------------------------
def get_secret(key: str, default: str = "") -> str:
    """Reads a secret from Streamlit Cloud st.secrets first, then OS environment variables."""
    if HAS_STREAMLIT:
        try:
            if key in st.secrets:
                val = st.secrets[key]
                return str(val).strip()
        except Exception:
            pass
    return os.environ.get(key, default).strip()


def find_reference_dxf() -> Optional[str]:
    """Locates the active reference DXF (teacher-uploaded first, else bundled v3 DXF)."""
    if ACTIVE_DXF_PATH.exists():
        return str(ACTIVE_DXF_PATH)
    for pattern in ("isometric_prism_on_cube-v3.dxf", "*.dxf"):
        matches = list(BASE_DIR.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def now_ist_str() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")


# ===========================================================================
# PART 1: SQLITE GRADEBOOK STORAGE & EXCEL/CSV EXPORT
# ===========================================================================
def init_db(db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submitted_at TEXT NOT NULL,
                full_name TEXT NOT NULL,
                u_id TEXT NOT NULL,
                email TEXT NOT NULL,
                scan_filename TEXT NOT NULL,
                grading_mode TEXT NOT NULL,
                detected_scale TEXT NOT NULL,
                effective_iso_factor REAL NOT NULL,
                deskew_deg REAL NOT NULL,
                m_scale REAL NOT NULL,
                m_cube REAL NOT NULL,
                m_prism REAL NOT NULL,
                m_hidden REAL NOT NULL,
                m_iso_scale_vlm REAL,
                m_dimensions REAL NOT NULL,
                auto_total REAL NOT NULL,
                max_marks REAL NOT NULL,
                approved_marks REAL NOT NULL,
                faculty_remark TEXT DEFAULT '',
                cv_report_json TEXT NOT NULL,
                vlm_audit_json TEXT,
                rubric_json TEXT NOT NULL,
                orig_image_blob BLOB NOT NULL,
                overlay_image_blob BLOB NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def load_settings(db_path: Path = DB_PATH) -> Dict[str, Any]:
    init_db(db_path)
    settings = dict(DEFAULT_SETTINGS)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT key, value_json FROM app_settings").fetchall()
    for k, v_json in rows:
        try:
            settings[k] = json.loads(v_json)
        except Exception:
            pass
    return settings


def save_settings(settings: Dict[str, Any], db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for k, v in settings.items():
            conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value_json) VALUES (?, ?)",
                (k, json.dumps(v)),
            )
        conn.commit()


def save_submission(
    full_name: str,
    u_id: str,
    email: str,
    scan_filename: str,
    cv_report: Dict,
    vlm_dict: Optional[Dict],
    rubric: Dict,
    orig_bytes: bytes,
    overlay_bytes: bytes,
    db_path: Path = DB_PATH,
) -> int:
    init_db(db_path)
    submitted_at = now_ist_str()
    grading_mode = "Hybrid CV + VLM" if vlm_dict is not None else "OpenCV Only"
    sel_grp = cv_report["scale_detection"]["selected_group"]
    detected_scale = (
        "Projection (0.8165)" if "PROJECTION" in sel_grp
        else ("View (1.0)" if "VIEW" in sel_grp else "Proportional")
    )

    row_map = {r["Criterion"]: r["Marks"] for r in rubric["rows"]}
    m_scale = float(row_map.get("Scale Accuracy (Projection 0.8165 vs View 1.0)", 0.0))
    m_cube = float(row_map.get("Bottom Cube Geometry (9 visible edges)", 0.0))
    m_prism = float(row_map.get("Top Square Prism Geometry (9 visible edges)", 0.0))
    m_hidden = float(row_map.get("Hidden Edges & Visibility Policy", 0.0))
    m_iso_scale_vlm = (
        float(row_map["Graphical Isometric Scale Construction"])
        if "Graphical Isometric Scale Construction" in row_map
        else None
    )
    m_dimensions = float(
        row_map.get(
            "Dimensioning, Arrowheads & Labeling",
            row_map.get("Dimension Lines (Fixed-Offset Check)", 0.0),
        )
    )

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO submissions (
                submitted_at, full_name, u_id, email, scan_filename, grading_mode,
                detected_scale, effective_iso_factor, deskew_deg,
                m_scale, m_cube, m_prism, m_hidden, m_iso_scale_vlm, m_dimensions,
                auto_total, max_marks, approved_marks, faculty_remark,
                cv_report_json, vlm_audit_json, rubric_json,
                orig_image_blob, overlay_image_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submitted_at,
                full_name.strip(),
                u_id.strip(),
                email.strip(),
                scan_filename,
                grading_mode,
                detected_scale,
                float(cv_report["scale_detection"]["effective_iso_factor"]),
                float(cv_report["calibration"]["deskew_angle_deg"]),
                m_scale,
                m_cube,
                m_prism,
                m_hidden,
                m_iso_scale_vlm,
                m_dimensions,
                float(rubric["total"]),
                float(rubric["max"]),
                float(rubric["total"]),
                "",
                json.dumps(cv_report),
                json.dumps(vlm_dict) if vlm_dict is not None else None,
                json.dumps(rubric),
                orig_bytes,
                overlay_bytes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def fetch_gradebook_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    query = """
        SELECT
            id AS "ID",
            submitted_at AS "Submitted At",
            u_id AS "U_ID",
            full_name AS "Full Name",
            email AS "Email",
            grading_mode AS "Mode",
            detected_scale AS "Detected Scale",
            effective_iso_factor AS "Iso Factor",
            deskew_deg AS "Deskew (deg)",
            m_scale AS "Scale Marks",
            m_cube AS "Cube Marks",
            m_prism AS "Prism Marks",
            m_hidden AS "Hidden Edges Marks",
            m_iso_scale_vlm AS "Iso Scale (VLM)",
            m_dimensions AS "Dimensions Marks",
            auto_total AS "Auto Total",
            approved_marks AS "Approved Marks",
            max_marks AS "Max Marks",
            faculty_remark AS "Faculty Remark"
        FROM submissions
        ORDER BY id DESC
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def fetch_submission_detail(sub_id: int, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM submissions WHERE id = ?", (sub_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["cv_report"] = json.loads(d["cv_report_json"])
    d["vlm_audit"] = json.loads(d["vlm_audit_json"]) if d["vlm_audit_json"] else None
    d["rubric"] = json.loads(d["rubric_json"])
    return d


def update_faculty_override(sub_id: int, approved_marks: float, faculty_remark: str, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE submissions SET approved_marks = ?, faculty_remark = ? WHERE id = ?",
            (round(float(approved_marks), 2), faculty_remark.strip(), int(sub_id)),
        )
        conn.commit()


def delete_submission(sub_id: int, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM submissions WHERE id = ?", (int(sub_id),))
        conn.commit()


def build_excel_bytes(df: pd.DataFrame) -> bytes:
    """Exports the tabulated rubric gradebook to a clean, formatted Excel (.xlsx) workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Rubric Gradebook")

        # Add a second sheet with class-level summary stats
        if not df.empty:
            summary_df = pd.DataFrame([
                {"Metric": "Total Submissions", "Value": len(df)},
                {"Metric": "Average Approved Score", "Value": round(float(df["Approved Marks"].mean()), 2)},
                {"Metric": "Highest Approved Score", "Value": round(float(df["Approved Marks"].max()), 2)},
                {"Metric": "Lowest Approved Score", "Value": round(float(df["Approved Marks"].min()), 2)},
                {"Metric": "Projection (0.8165) Count", "Value": int((df["Detected Scale"] == "Projection (0.8165)").sum())},
                {"Metric": "True-Scale View (1.0) Errors", "Value": int((df["Detected Scale"] == "View (1.0)").sum())},
            ])
        else:
            summary_df = pd.DataFrame([{"Metric": "Total Submissions", "Value": 0}])
        summary_df.to_excel(writer, index=False, sheet_name="Class Summary")

        if HAS_OPENPYXL:
            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            for ws_name in writer.sheets:
                ws = writer.sheets[ws_name]
                ws.freeze_panes = "A2"
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                for col_cells in ws.columns:
                    col_letter = col_cells[0].column_letter
                    max_len = max(len(str(c.value or "")) for c in col_cells)
                    ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 42)
    return buf.getvalue()


def maybe_sync_to_google_sheet(sub_id: int, db_path: Path = DB_PATH) -> Optional[str]:
    """
    Optional cloud persistence: if `gcp_service_account` and `GSHEET_URL` exist in st.secrets
    and `gspread` is installed, appends the new submission row to the teacher's Google Sheet.
    """
    if not HAS_STREAMLIT:
        return None
    try:
        if "gcp_service_account" not in st.secrets or "GSHEET_URL" not in st.secrets:
            return None
        import gspread
    except Exception:
        return None

    try:
        df = fetch_gradebook_df(db_path)
        row_df = df[df["ID"] == sub_id]
        if row_df.empty:
            return None
        gc = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        sh = gc.open_by_url(str(st.secrets["GSHEET_URL"]))
        ws = sh.sheet1
        if not ws.get_all_values():
            ws.append_row(list(row_df.columns))
        values = ["" if pd.isna(v) else (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, (np.integer,)) else v))
                  for v in row_df.iloc[0].tolist()]
        ws.append_row(values)
        return "Synced to Google Sheet"
    except Exception as exc:
        return f"Google Sheet sync skipped ({exc})"


# ===========================================================================
# PART 2: BORDERLESS DXF + OPENCV GEOMETRIC GRADING ENGINE
# ===========================================================================
def load_dxf_layers(dxf_path: str) -> Dict[str, List[Segment]]:
    """Loads LINE segments grouped by DXF layer name in CAD mm (y-up)."""
    layers: Dict[str, List[Segment]] = {}

    if ezdxf is not None:
        doc = ezdxf.readfile(dxf_path)
        for entity in doc.modelspace().query("LINE"):
            lyr = entity.dxf.layer
            p1 = (float(entity.dxf.start.x), float(entity.dxf.start.y))
            p2 = (float(entity.dxf.end.x), float(entity.dxf.end.y))
            layers.setdefault(lyr, []).append((p1, p2))
        return layers

    # Fallback ASCII R12 DXF LINE parser
    with open(dxf_path, "r", encoding="ascii", errors="ignore") as f:
        raw = [line.strip() for line in f]
    pairs = [(int(raw[i]), raw[i + 1]) for i in range(0, len(raw) - 1, 2)]
    i, in_entities = 0, False
    while i < len(pairs):
        code, val = pairs[i]
        if code == 2 and val == "ENTITIES":
            in_entities = True
        elif in_entities and code == 0 and val == "ENDSEC":
            break
        elif in_entities and code == 0 and val == "LINE":
            d = {}
            i += 1
            while i < len(pairs) and pairs[i][0] != 0:
                d[pairs[i][0]] = pairs[i][1]
                i += 1
            lyr = d.get(8, "0")
            p1 = (float(d[10]), float(d[20]))
            p2 = (float(d[11]), float(d[21]))
            layers.setdefault(lyr, []).append((p1, p2))
            continue
        i += 1
    return layers


def resolve_px_per_mm(
    image_path: str,
    img_width_px: int,
    img_height_px: int,
    dpi: Optional[float] = None,
    page_width_mm: Optional[float] = None,
    page_height_mm: Optional[float] = None,
) -> Tuple[float, str]:
    if dpi is not None and dpi > 0:
        return dpi / 25.4, f"explicit_dpi ({dpi:.1f} DPI)"
    if page_height_mm is not None and page_height_mm > 0:
        return img_height_px / page_height_mm, f"page_height_mm ({page_height_mm:.1f} mm)"
    if page_width_mm is not None and page_width_mm > 0:
        return img_width_px / page_width_mm, f"page_width_mm ({page_width_mm:.1f} mm)"
    try:
        with Image.open(image_path) as im:
            meta_dpi = im.info.get("dpi")
            if meta_dpi and meta_dpi[0] and meta_dpi[0] >= 50:
                return float(meta_dpi[0]) / 25.4, f"image_metadata ({float(meta_dpi[0]):.1f} DPI)"
    except Exception:
        pass
    return 200.0 / 25.4, "assumed_default_200_dpi"


def extract_ink_masks(bgr: np.ndarray, px_per_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_k = max(31, int(round(px_per_mm * 8.0)) | 1)
    bg = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    darkness = np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)

    all_ink = (darkness >= 18).astype(np.uint8) * 255
    all_ink = cv2.medianBlur(all_ink, 3)

    ink_vals = darkness[all_ink > 0]
    if ink_vals.size > 50:
        p50 = float(np.percentile(ink_vals, 50))
        p85 = float(np.percentile(ink_vals, 85))
        dark_thresh = max(45.0, min(110.0, 0.45 * p50 + 0.55 * p85)) if (p85 - p50) > 25.0 else max(35.0, 0.5 * p85)
    else:
        dark_thresh = 50.0

    dark_ink = (darkness >= dark_thresh).astype(np.uint8) * 255
    dark_ink = cv2.medianBlur(dark_ink, 3)
    return all_ink, dark_ink


def auto_deskew_isometric(bgr: np.ndarray, dark_ink: np.ndarray, px_per_mm: float) -> Tuple[np.ndarray, float]:
    min_len = max(15, int(round(px_per_mm * 8.0)))
    max_gap = max(3, int(round(px_per_mm * 1.5)))
    lines = cv2.HoughLinesP(dark_ink, 1, np.pi / 360.0, threshold=30, minLineLength=min_len, maxLineGap=max_gap)

    if lines is None or lines.size < 12:
        return bgr.copy(), 0.0

    residuals, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        ang = math.degrees(math.atan2(-dy, dx)) % 180.0
        res = ((ang - 30.0 + 30.0) % 60.0) - 30.0
        if abs(res) <= 15.0:
            residuals.append(res)
            weights.append(length)

    if not residuals:
        return bgr.copy(), 0.0

    res_arr = np.array(residuals, dtype=np.float64)
    w_arr = np.array(weights, dtype=np.float64)
    bins = np.arange(-15.25, 15.75, 0.5)
    hist, edges = np.histogram(res_arr, bins=bins, weights=w_arr)
    peak_center = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    inliers = np.abs(res_arr - peak_center) <= 2.0
    tilt_deg = float(np.sum(res_arr[inliers] * w_arr[inliers]) / np.sum(w_arr[inliers]))

    if abs(tilt_deg) < 0.15:
        return bgr.copy(), 0.0

    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -tilt_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    deskewed = cv2.warpAffine(bgr, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return deskewed, round(tilt_deg, 3)


def get_bounds(segments: List[Segment]) -> Tuple[float, float, float, float]:
    xs = [p[0] for s in segments for p in s]
    ys = [p[1] for s in segments for p in s]
    return min(xs), min(ys), max(xs), max(ys)


def cad_to_px(
    pt: Tuple[float, float],
    origin_px: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    px_per_mm: float,
    scale_mult: float,
    pad_px: int,
) -> Tuple[float, float]:
    xmin, _, _, ymax = bounds
    eff = px_per_mm * scale_mult
    u = origin_px[0] + pad_px + (pt[0] - xmin) * eff
    v = origin_px[1] + pad_px + (ymax - pt[1]) * eff
    return (u, v)


def rasterize_template(
    segments: List[Segment],
    bounds: Tuple[float, float, float, float],
    px_per_mm: float,
    scale_mult: float,
    pad_px: int = 4,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    eff = px_per_mm * scale_mult
    w = int(math.ceil((xmax - xmin) * eff)) + 2 * pad_px + 1
    h = int(math.ceil((ymax - ymin) * eff)) + 2 * pad_px + 1
    tmpl = np.zeros((h, w), dtype=np.uint8)
    for p1, p2 in segments:
        u1, v1 = cad_to_px(p1, (0.0, 0.0), bounds, px_per_mm, scale_mult, pad_px)
        u2, v2 = cad_to_px(p2, (0.0, 0.0), bounds, px_per_mm, scale_mult, pad_px)
        cv2.line(tmpl, (int(round(u1)), int(round(v1))), (int(round(u2)), int(round(v2))), 255, thickness=1)
    return (tmpl > 0).astype(np.float32)


def align_dxf_dual_scale(
    visible_segments: List[Segment],
    dark_ink: np.ndarray,
    px_per_mm: float,
    tol_mm: float = 2.0,
    unknown_dpi: bool = False,
) -> Dict:
    bounds = get_bounds(visible_segments)
    dt_px = cv2.distanceTransform(cv2.bitwise_not(dark_ink), cv2.DIST_L2, 5)
    dt_mm = dt_px / px_per_mm
    chamfer_map = np.minimum(dt_mm, 10.0).astype(np.float32)

    pad_px = 4
    img_h, img_w = chamfer_map.shape

    if unknown_dpi:
        candidate_groups = [("UNKNOWN_DPI_PROPORTIONAL", np.linspace(0.55, 1.85, 27).tolist())]
    else:
        jitter = [0.97, 0.985, 1.00, 1.015, 1.03]
        candidate_groups = [
            ("ISOMETRIC_PROJECTION_0.8165", [1.0 * j for j in jitter]),
            ("ISOMETRIC_VIEW_1.0000", [VIEW_OVER_PROJ * j for j in jitter]),
        ]

    group_best = {}
    overall_best = None

    for group_name, mults in candidate_groups:
        best_for_group = None
        for mult in mults:
            tmpl = rasterize_template(visible_segments, bounds, px_per_mm, mult, pad_px=pad_px)
            th, tw = tmpl.shape
            if th >= img_h or tw >= img_w or tmpl.sum() < 10:
                continue

            ds = 0.5 if max(img_h, img_w) > 1600 else 1.0
            if ds < 1.0:
                cm_s = cv2.resize(chamfer_map, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
                tm_s = cv2.resize(tmpl, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_LINEAR)
                tm_s = (tm_s > 0.2).astype(np.float32)
                if tm_s.shape[0] >= cm_s.shape[0] or tm_s.shape[1] >= cm_s.shape[1] or tm_s.sum() < 10:
                    continue
                resp = cv2.matchTemplate(cm_s, tm_s, cv2.TM_CCORR) / float(tm_s.sum())
                _, _, min_loc_s, _ = cv2.minMaxLoc(resp)
                cx, cy = int(round(min_loc_s[0] / ds)), int(round(min_loc_s[1] / ds))
                x0, y0 = max(0, cx - 6), max(0, cy - 6)
                x1, y1 = min(img_w, cx + tw + 6), min(img_h, cy + th + 6)
                roi = chamfer_map[y0:y1, x0:x1]
                if roi.shape[0] >= th and roi.shape[1] >= tw:
                    resp_fine = cv2.matchTemplate(roi, tmpl, cv2.TM_CCORR) / float(tmpl.sum())
                    _, min_val, min_loc_f, _ = cv2.minMaxLoc(resp_fine)
                    ox, oy = x0 + min_loc_f[0], y0 + min_loc_f[1]
                else:
                    ox, oy = min(cx, img_w - tw), min(cy, img_h - th)
                    min_val = float(np.sum(chamfer_map[oy:oy + th, ox:ox + tw] * tmpl) / tmpl.sum())
            else:
                resp = cv2.matchTemplate(chamfer_map, tmpl, cv2.TM_CCORR) / float(tmpl.sum())
                _, min_val, (ox, oy), _ = cv2.minMaxLoc(resp)

            patch_dt = dt_mm[oy:oy + th, ox:ox + tw]
            tmpl_bool = tmpl > 0.5
            fwd_cov = float(np.mean(patch_dt[tmpl_bool] <= tol_mm))

            dil_r = max(1, int(round(tol_mm * px_per_mm)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_r + 1, 2 * dil_r + 1))
            tmpl_dil = cv2.dilate((tmpl_bool.astype(np.uint8) * 255), kernel) > 0
            roi_dark = dark_ink[oy:oy + th, ox:ox + tw] > 0
            rev_prec = float(np.mean(tmpl_dil[roi_dark])) if np.any(roi_dark) else 0.0

            score = 0.65 * fwd_cov + 0.35 * rev_prec - 0.04 * float(min_val)
            cand = {
                "group": group_name,
                "scale_mult": round(mult, 5),
                "effective_iso_factor": round(ISO_FACTOR * mult, 4),
                "origin_px": (int(ox), int(oy)),
                "pad_px": pad_px,
                "bounds": bounds,
                "mean_chamfer_mm": round(float(min_val), 3),
                "forward_coverage": round(fwd_cov, 4),
                "reverse_precision": round(rev_prec, 4),
                "alignment_score": round(score, 4),
            }
            if best_for_group is None or cand["alignment_score"] > best_for_group["alignment_score"]:
                best_for_group = cand

        if best_for_group is not None:
            group_best[group_name] = best_for_group
            if overall_best is None or best_for_group["alignment_score"] > overall_best["alignment_score"]:
                overall_best = best_for_group

    if overall_best is None:
        raise RuntimeError("Alignment failed: template exceeded scan dimensions. Check page calibration.")

    return {"best": overall_best, "candidates": group_best}


def sample_segment_px(
    seg: Segment,
    align: Dict,
    px_per_mm: float,
    step_mm: float = 0.5,
    trim_ends_mm: float = 0.0,
) -> np.ndarray:
    (x1, y1), (x2, y2) = seg
    seg_len_mm = math.hypot(x2 - x1, y2 - y1) * align["scale_mult"]
    if seg_len_mm <= 1e-3:
        return np.empty((0, 2), dtype=np.float32)

    eff_trim = min(trim_ends_mm, 0.30 * seg_len_mm)
    t0 = eff_trim / seg_len_mm
    t1 = 1.0 - t0
    n_pts = max(5, int(math.ceil((seg_len_mm - 2 * eff_trim) / step_mm)))
    ts = np.linspace(t0, t1, n_pts)

    pts = [
        cad_to_px(
            (x1 + t * (x2 - x1), y1 + t * (y2 - y1)),
            align["origin_px"], align["bounds"], px_per_mm, align["scale_mult"], align["pad_px"]
        )
        for t in ts
    ]
    return np.array(pts, dtype=np.float32)


def eval_points_on_dt(pts: np.ndarray, dt_mm: np.ndarray, tol_mm: float) -> Tuple[float, float]:
    h, w = dt_mm.shape
    if pts.shape[0] == 0:
        return 0.0, 99.0
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    dists = dt_mm[ys, xs]
    return float(np.mean(dists <= tol_mm)), float(np.mean(dists))


def grade_scan(
    dxf_path: str,
    scan_path: str,
    dpi: Optional[float] = None,
    page_width_mm: Optional[float] = None,
    page_height_mm: Optional[float] = None,
    unknown_dpi: bool = False,
    hidden_policy: str = "omit",
    tol_mm: float = 2.5,
    out_overlay: Optional[str] = None,
    out_json: Optional[str] = None,
) -> Dict:
    layers = load_dxf_layers(dxf_path)
    bgr_raw = cv2.imread(scan_path)
    if bgr_raw is None:
        raise FileNotFoundError(f"Could not read scan image: {scan_path}")

    px_per_mm, dpi_source = resolve_px_per_mm(
        scan_path, bgr_raw.shape[1], bgr_raw.shape[0],
        dpi=dpi, page_width_mm=page_width_mm, page_height_mm=page_height_mm,
    )

    _, dark_raw = extract_ink_masks(bgr_raw, px_per_mm)
    bgr, tilt_deg = auto_deskew_isometric(bgr_raw, dark_raw, px_per_mm)
    all_ink, dark_ink = extract_ink_masks(bgr, px_per_mm)

    dt_dark_mm = cv2.distanceTransform(cv2.bitwise_not(dark_ink), cv2.DIST_L2, 5) / px_per_mm
    dt_all_mm = cv2.distanceTransform(cv2.bitwise_not(all_ink), cv2.DIST_L2, 5) / px_per_mm

    vis_cube = layers.get("VISIBLE_CUBE", [])
    vis_prism = layers.get("VISIBLE_PRISM", [])
    align_res = align_dxf_dual_scale(
        vis_cube + vis_prism, dark_ink, px_per_mm, tol_mm=tol_mm, unknown_dpi=unknown_dpi
    )
    align = align_res["best"]

    def score_visible_layer(segs: List[Segment], layer_name: str) -> Dict:
        details, passed, faint_only = [], 0, 0
        for idx, seg in enumerate(segs):
            pts = sample_segment_px(seg, align, px_per_mm, step_mm=0.5)
            cov_dark, mean_dark = eval_points_on_dt(pts, dt_dark_mm, tol_mm)
            cov_all, _ = eval_points_on_dt(pts, dt_all_mm, tol_mm)
            if cov_dark >= 0.75:
                status = "PASS"; passed += 1
            elif cov_all >= 0.75:
                status = "FAINT"; faint_only += 1
            else:
                status = "MISSING"
            details.append({
                "index": idx, "status": status,
                "dark_coverage": round(cov_dark, 3), "all_ink_coverage": round(cov_all, 3),
                "mean_error_mm": round(mean_dark, 2), "seg": seg,
            })
        score_frac = (passed + 0.5 * faint_only) / max(1, len(segs))
        return {
            "layer": layer_name, "total_edges": len(segs), "passed": passed,
            "faint_only": faint_only, "missing": len(segs) - passed - faint_only,
            "score_fraction": round(score_frac, 4), "edges": details,
        }

    cube_res = score_visible_layer(vis_cube, "VISIBLE_CUBE")
    prism_res = score_visible_layer(vis_prism, "VISIBLE_PRISM")

    hidden_segs = [(lyr, seg) for lyr in ("HIDDEN_CUBE", "HIDDEN_PRISM") for seg in layers.get(lyr, [])]
    hidden_details, hidden_violations = [], 0
    for idx, (lyr, seg) in enumerate(hidden_segs):
        pts = sample_segment_px(seg, align, px_per_mm, step_mm=0.5, trim_ends_mm=2.5)
        cov_dark, _ = eval_points_on_dt(pts, dt_dark_mm, min(1.2, tol_mm))
        cov_all, _ = eval_points_on_dt(pts, dt_all_mm, tol_mm)
        if hidden_policy == "omit":
            if cov_dark >= 0.45:
                status = "UNERASED_DARK_HIDDEN"; hidden_violations += 1
            else:
                status = "CLEAN"
        else:
            if 0.30 <= cov_all <= 0.90:
                status = "DASHED_OK"
            else:
                status = "MISSING_OR_SOLID_HIDDEN"; hidden_violations += 1
        hidden_details.append({
            "index": idx, "layer": lyr, "status": status,
            "dark_presence": round(cov_dark, 3), "all_ink_presence": round(cov_all, 3), "seg": seg,
        })

    hidden_score_frac = max(0.0, 1.0 - (hidden_violations / max(1, len(hidden_segs))))

    dim_segs = layers.get("DIMENSIONS", [])
    dim_details, dim_hits = [], 0
    for idx, seg in enumerate(dim_segs):
        pts = sample_segment_px(seg, align, px_per_mm, step_mm=0.75)
        cov_all, mean_all = eval_points_on_dt(pts, dt_all_mm, tol_mm * 1.75)
        ok = cov_all >= 0.55
        if ok:
            dim_hits += 1
        dim_details.append({
            "index": idx, "status": "MATCHED" if ok else "UNMATCHED_OR_RELOCATED",
            "coverage": round(cov_all, 3), "mean_error_mm": round(mean_all, 2), "seg": seg,
        })
    dim_score_frac = dim_hits / max(1, len(dim_segs)) if dim_segs else 0.0

    if unknown_dpi:
        scale_verdict = "UNVERIFIED (DPI unknown - graded on proportions only)"
        scale_marks = RUBRIC_WEIGHTS["scale_projection_vs_view"] * 0.5
    elif align["group"] == "ISOMETRIC_PROJECTION_0.8165":
        scale_verdict = "ISOMETRIC_PROJECTION (0.8165) - CORRECT"
        scale_marks = RUBRIC_WEIGHTS["scale_projection_vs_view"]
    else:
        scale_verdict = "ISOMETRIC_VIEW (1.0000) - TRUE SCALE USED INSTEAD OF ISOMETRIC SCALE"
        scale_marks = 0.0

    marks = {
        "scale_projection_vs_view": round(scale_marks, 2),
        "visible_cube": round(RUBRIC_WEIGHTS["visible_cube"] * cube_res["score_fraction"], 2),
        "visible_prism": round(RUBRIC_WEIGHTS["visible_prism"] * prism_res["score_fraction"], 2),
        "hidden_edges": round(RUBRIC_WEIGHTS["hidden_edges"] * hidden_score_frac, 2),
        "dimensions": round(RUBRIC_WEIGHTS["dimensions"] * dim_score_frac, 2),
    }
    total_marks = round(sum(marks.values()), 2)
    max_marks = round(sum(RUBRIC_WEIGHTS.values()), 2)

    feedback = []
    if align["group"] == "ISOMETRIC_VIEW_1.0000":
        feedback.append(
            "Scale error: Drawing matches 1:1 true lengths (Isometric View) rather than foreshortened "
            "Isometric Projection lengths (x 0.8165). Use an isometric scale (60->49 mm, 50->41 mm, 40->33 mm)."
        )
    if cube_res["missing"] > 0:
        feedback.append(f"Bottom cube: {cube_res['missing']} of {cube_res['total_edges']} visible edges missing or misaligned.")
    if cube_res["faint_only"] > 0:
        feedback.append(f"Bottom cube: {cube_res['faint_only']} visible edge(s) drawn too faintly (darken with HB/B pencil).")
    if prism_res["missing"] > 0:
        feedback.append(f"Top square prism: {prism_res['missing']} of {prism_res['total_edges']} visible edges missing or misaligned (check centering/height).")
    if prism_res["faint_only"] > 0:
        feedback.append(f"Top square prism: {prism_res['faint_only']} visible edge(s) drawn too faintly.")
    if hidden_violations > 0:
        if hidden_policy == "omit":
            feedback.append(
                f"Hidden edges: {hidden_violations} occluded back edge(s) were left drawn darkly; "
                "per BIS SP 46:2003, omit hidden edges in final isometric projections."
            )
        else:
            feedback.append(f"Hidden edges: {hidden_violations} required dashed hidden edge(s) missing or drawn solid.")
    if dim_score_frac < 0.6:
        feedback.append(
            "Dimensioning: Some expected dimension/extension lines were not matched at reference offsets "
            "(verify numerals, 3:1 arrowheads, and alternate placement via VLM audit)."
        )
    if not feedback:
        feedback.append("Excellent isometric projection: accurate 0.8165 scaling, clean visibility, and complete edges.")

    overlay = bgr.copy()
    def draw_seg(seg: Segment, color: Tuple[int, int, int], thickness: int = 2, dashed: bool = False):
        p1 = cad_to_px(seg[0], align["origin_px"], align["bounds"], px_per_mm, align["scale_mult"], align["pad_px"])
        p2 = cad_to_px(seg[1], align["origin_px"], align["bounds"], px_per_mm, align["scale_mult"], align["pad_px"])
        pt1, pt2 = (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))
        if not dashed:
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
        else:
            dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            steps = max(2, int(dist / 8))
            for k in range(0, steps, 2):
                a, b = k / steps, min(1.0, (k + 1) / steps)
                q1 = (int(round(pt1[0] + a * (pt2[0] - pt1[0]))), int(round(pt1[1] + a * (pt2[1] - pt1[1]))))
                q2 = (int(round(pt1[0] + b * (pt2[0] - pt1[0]))), int(round(pt1[1] + b * (pt2[1] - pt1[1]))))
                cv2.line(overlay, q1, q2, color, thickness, cv2.LINE_AA)

    for item in dim_details:
        if item["status"] == "MATCHED":
            draw_seg(item["seg"], (235, 130, 30), thickness=2)
        else:
            draw_seg(item["seg"], (220, 180, 120), thickness=1, dashed=True)

    for item in hidden_details:
        if item["status"] in ("UNERASED_DARK_HIDDEN", "MISSING_OR_SOLID_HIDDEN"):
            draw_seg(item["seg"], (0, 140, 255), thickness=3, dashed=True)

    for res in (cube_res, prism_res):
        for item in res["edges"]:
            if item["status"] == "PASS":
                draw_seg(item["seg"], (40, 180, 40), thickness=3)
            elif item["status"] == "FAINT":
                draw_seg(item["seg"], (0, 180, 255), thickness=3)
            else:
                draw_seg(item["seg"], (30, 30, 230), thickness=3)

    banner_h = 74
    canvas = np.full((overlay.shape[0] + banner_h, overlay.shape[1], 3), 248, dtype=np.uint8)
    canvas[banner_h:, :] = overlay
    cv2.line(canvas, (0, banner_h - 1), (canvas.shape[1], banner_h - 1), (200, 200, 200), 1)
    cv2.putText(
        canvas,
        f"CV Score: {total_marks:.1f}/{max_marks:.1f}  |  Scale: {scale_verdict}",
        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Deskew: {tilt_deg:+.2f} deg  |  Cube: {cube_res['passed']}/{cube_res['total_edges']}  |  "
        f"Prism: {prism_res['passed']}/{prism_res['total_edges']}  |  Hidden Issues: {hidden_violations}  |  "
        f"Green=OK  Red=Missing  Amber=Faint",
        (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (70, 70, 70), 1, cv2.LINE_AA,
    )

    if out_overlay:
        cv2.imwrite(out_overlay, canvas)

    def strip_segs(items):
        return [{k: v for k, v in d.items() if k != "seg"} for d in items]

    report = {
        "scan_path": str(scan_path),
        "dxf_path": str(dxf_path),
        "calibration": {
            "px_per_mm": round(px_per_mm, 3),
            "source": dpi_source,
            "deskew_angle_deg": tilt_deg,
            "tolerance_mm": tol_mm,
        },
        "scale_detection": {
            "verdict": scale_verdict,
            "selected_group": align["group"],
            "effective_iso_factor": align["effective_iso_factor"],
            "mean_chamfer_mm": align["mean_chamfer_mm"],
            "candidates": {
                k: {
                    "effective_iso_factor": v["effective_iso_factor"],
                    "forward_coverage": v["forward_coverage"],
                    "reverse_precision": v["reverse_precision"],
                    "mean_chamfer_mm": v["mean_chamfer_mm"],
                    "alignment_score": v["alignment_score"],
                }
                for k, v in align_res["candidates"].items()
            },
        },
        "layer_results": {
            "VISIBLE_CUBE": {**cube_res, "edges": strip_segs(cube_res["edges"])},
            "VISIBLE_PRISM": {**prism_res, "edges": strip_segs(prism_res["edges"])},
            "HIDDEN_EDGES": {
                "policy": hidden_policy,
                "total": len(hidden_details),
                "violations": hidden_violations,
                "score_fraction": round(hidden_score_frac, 4),
                "edges": strip_segs(hidden_details),
            },
            "DIMENSIONS": {
                "matched": dim_hits,
                "total": len(dim_details),
                "score_fraction": round(dim_score_frac, 4),
                "note": "Complementary VLM audit verifies dimension numerals, arrowheads, and Isometric Scale.",
            },
        },
        "rubric_marks": marks,
        "total_marks": total_marks,
        "max_marks": max_marks,
        "feedback": feedback,
    }

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


# ===========================================================================
# PART 3: GEMINI VLM NOTATION & ISOMETRIC SCALE AUDIT
# ===========================================================================
if HAS_PYDANTIC:
    class IsometricScaleAudit(BaseModel):
        present: bool = Field(description="True if a graphical Isometric Scale (30 deg & 45 deg lines) is drawn.")
        has_30_and_45_rays: bool = Field(description="True if both 30 deg and 45 deg rays are drawn and/or labeled.")
        requisite_lengths_marked: List[str] = Field(description="Dimension values marked on the scale (e.g., ['40', '50', '60']).")
        score_out_of_1_5: float = Field(ge=0.0, le=1.5, description="Marks for graphical Isometric Scale (0.0 to 1.5).")
        notes: str = Field(description="Concise technical observation on the student's isometric scale.")

    class DimensioningAudit(BaseModel):
        dimensions_found: List[str] = Field(description="Handwritten dimensions seen on the solid (e.g., ['Iso 60']).")
        missing_dimensions: List[str] = Field(description="Requisite dimensions missing (expected: cube 60, prism base 40, prism height 50).")
        arrowheads_and_conventions: str = Field(description="Brief evaluation of 3:1 filled arrowheads, alignment, and extension lines.")
        score_out_of_1_0: float = Field(ge=0.0, le=1.0, description="Marks for dimensioning, arrowheads, and labeling (0.0 to 1.0).")
        notes: str = Field(description="Concise technical observation on dimensioning and notation.")

    class VLMAuditResult(BaseModel):
        isometric_scale: IsometricScaleAudit
        dimensioning_and_notation: DimensioningAudit
        neatness_and_line_weights: str = Field(description="Brief comment on line-weight contrast and drawing neatness.")
        combined_student_feedback: List[str] = Field(
            description="3 to 5 clear, actionable feedback bullets merging OpenCV geometry findings with the visual notation audit."
        )


def run_gemini_vlm_audit(
    model_id: str,
    problem_statement: str,
    original_scan_path: str,
    overlay_path: str,
    cv_report: Dict,
    api_key_override: str = "",
) -> Dict:
    """Calls Gemini via `google-genai` SDK using st.secrets / GEMINI_API_KEY."""
    if not HAS_GENAI or not HAS_PYDANTIC:
        raise RuntimeError("Install missing VLM packages: pip install google-genai pydantic")

    key = api_key_override.strip() or get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in Streamlit Secrets or environment variables.")

    client = genai.Client(api_key=key)

    orig_bytes = Path(original_scan_path).read_bytes()
    overlay_bytes = Path(overlay_path).read_bytes()
    orig_mime = "image/png" if original_scan_path.lower().endswith(".png") else "image/jpeg"

    cv_summary = {
        "scale_detection": cv_report["scale_detection"]["verdict"],
        "effective_iso_factor": cv_report["scale_detection"]["effective_iso_factor"],
        "deskew_angle_deg": cv_report["calibration"]["deskew_angle_deg"],
        "visible_cube": {
            "passed": cv_report["layer_results"]["VISIBLE_CUBE"]["passed"],
            "faint_only": cv_report["layer_results"]["VISIBLE_CUBE"]["faint_only"],
            "missing": cv_report["layer_results"]["VISIBLE_CUBE"]["missing"],
            "total": cv_report["layer_results"]["VISIBLE_CUBE"]["total_edges"],
        },
        "visible_prism": {
            "passed": cv_report["layer_results"]["VISIBLE_PRISM"]["passed"],
            "faint_only": cv_report["layer_results"]["VISIBLE_PRISM"]["faint_only"],
            "missing": cv_report["layer_results"]["VISIBLE_PRISM"]["missing"],
            "total": cv_report["layer_results"]["VISIBLE_PRISM"]["total_edges"],
        },
        "hidden_edges": {
            "policy": cv_report["layer_results"]["HIDDEN_EDGES"]["policy"],
            "violations": cv_report["layer_results"]["HIDDEN_EDGES"]["violations"],
        },
        "cv_feedback": cv_report["feedback"],
    }

    prompt = f"""You are an Engineering Graphics professor auditing a student's hand-drawn solution under BIS SP 46:2003.

Problem Statement:
{problem_statement}

You are given TWO images and a deterministic OpenCV geometric report:
- Image 1: The student's original scanned answer sheet.
- Image 2: The OpenCV CAD-DXF verification overlay (Green = matched dark edge, Amber = faint edge, Red = missing/misaligned edge, Orange dashed = hidden-edge issue).

Deterministic OpenCV Geometric Measurements (TRUST THESE FOR EXACT GEOMETRY & SCALE):
{json.dumps(cv_summary, indent=2)}

Inspect what OpenCV cannot reliably judge on Image 1, grounded in Image 2 and the JSON above:
1. Isometric Scale Construction (0.0 to 1.5 marks): Check if the student drew a graphical isometric scale (30 deg & 45 deg lines, vertical projections, and requisite markings such as 40, 50, 60 / Iso 40, Iso 50, Iso 60). Note any labeling errors (e.g., 75 deg instead of 45 deg).
2. Dimensioning, Arrowheads & Labeling (0.0 to 1.0 marks): Check handwritten dimension numerals on the solid (expected: cube 60, prism base 40, prism height 50; 'Iso 60' style is acceptable if consistent), 3:1 filled arrowheads, title ('Isometric Projection'), and corner labels.
3. Synthesize 3-5 concise, constructive student feedback bullets combining both OpenCV geometric defects and your notation/scale audit.

Return strictly valid JSON matching the requested schema.
"""

    response = client.models.generate_content(
        model=model_id,
        contents=[
            prompt,
            genai_types.Part.from_bytes(data=orig_bytes, mime_type=orig_mime),
            genai_types.Part.from_bytes(data=overlay_bytes, mime_type="image/png"),
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VLMAuditResult,
            temperature=0.1,
        ),
    )
    return VLMAuditResult.model_validate_json(response.text).model_dump()


def compute_hybrid_rubric(cv_report: Dict, vlm_dict: Optional[Dict]) -> Dict:
    """Combines CV geometry (7.5 marks) + Gemini VLM audit (2.5 marks) = 10.0 marks."""
    if vlm_dict is None:
        rows = [
            {"Criterion": "Scale Accuracy (Projection 0.8165 vs View 1.0)", "Engine": "OpenCV + DXF", "Marks": cv_report["rubric_marks"]["scale_projection_vs_view"], "Max": 2.0},
            {"Criterion": "Bottom Cube Geometry (9 visible edges)", "Engine": "OpenCV + DXF", "Marks": cv_report["rubric_marks"]["visible_cube"], "Max": 3.0},
            {"Criterion": "Top Square Prism Geometry (9 visible edges)", "Engine": "OpenCV + DXF", "Marks": cv_report["rubric_marks"]["visible_prism"], "Max": 3.0},
            {"Criterion": "Hidden Edges & Visibility Policy", "Engine": "OpenCV + DXF", "Marks": cv_report["rubric_marks"]["hidden_edges"], "Max": 1.0},
            {"Criterion": "Dimension Lines (Fixed-Offset Check)", "Engine": "OpenCV + DXF", "Marks": cv_report["rubric_marks"]["dimensions"], "Max": 1.0},
        ]
        return {"rows": rows, "total": cv_report["total_marks"], "max": cv_report["max_marks"]}

    scale_frac = cv_report["rubric_marks"]["scale_projection_vs_view"] / 2.0
    cube_frac = cv_report["layer_results"]["VISIBLE_CUBE"]["score_fraction"]
    prism_frac = cv_report["layer_results"]["VISIBLE_PRISM"]["score_fraction"]
    hidden_frac = cv_report["layer_results"]["HIDDEN_EDGES"]["score_fraction"]

    rows = [
        {"Criterion": "Scale Accuracy (Projection 0.8165 vs View 1.0)", "Engine": "OpenCV + DXF", "Marks": round(1.5 * scale_frac, 2), "Max": 1.5},
        {"Criterion": "Bottom Cube Geometry (9 visible edges)", "Engine": "OpenCV + DXF", "Marks": round(2.5 * cube_frac, 2), "Max": 2.5},
        {"Criterion": "Top Square Prism Geometry (9 visible edges)", "Engine": "OpenCV + DXF", "Marks": round(2.5 * prism_frac, 2), "Max": 2.5},
        {"Criterion": "Hidden Edges & Visibility Policy", "Engine": "OpenCV + DXF", "Marks": round(1.0 * hidden_frac, 2), "Max": 1.0},
        {"Criterion": "Graphical Isometric Scale Construction", "Engine": "Gemini VLM", "Marks": round(float(vlm_dict["isometric_scale"]["score_out_of_1_5"]), 2), "Max": 1.5},
        {"Criterion": "Dimensioning, Arrowheads & Labeling", "Engine": "Gemini VLM", "Marks": round(float(vlm_dict["dimensioning_and_notation"]["score_out_of_1_0"]), 2), "Max": 1.0},
    ]
    return {"rows": rows, "total": round(sum(r["Marks"] for r in rows), 2), "max": 10.0}


# ===========================================================================
# PART 4: STREAMLIT PORTAL (STUDENT SUBMISSION TAB + TEACHER DASHBOARD TAB)
# ===========================================================================
def render_result_details(
    orig_bytes: bytes,
    overlay_bytes: bytes,
    cv_rep: Dict,
    vlm_dict: Optional[Dict],
    rubric: Dict,
    approved_marks: Optional[float] = None,
) -> None:
    """Reusable visual report renderer for both Student feedback and Teacher inspection."""
    c1, c2, c3, c4 = st.columns(4)
    display_score = approved_marks if approved_marks is not None else rubric["total"]
    c1.metric("Score", f"{display_score:.2f} / {rubric['max']:.1f}")
    sel_grp = cv_rep["scale_detection"]["selected_group"]
    scale_short = (
        "Projection (0.8165)" if "PROJECTION" in sel_grp
        else ("View (1.0)" if "VIEW" in sel_grp else "Proportional")
    )
    c2.metric("Detected Scale", scale_short, f"factor {cv_rep['scale_detection']['effective_iso_factor']}")
    c3.metric("Auto-Deskew", f"{cv_rep['calibration']['deskew_angle_deg']:+.2f}°")
    cube_p = cv_rep["layer_results"]["VISIBLE_CUBE"]["passed"]
    prism_p = cv_rep["layer_results"]["VISIBLE_PRISM"]["passed"]
    c4.metric("Dark Visible Edges", f"{cube_p + prism_p} / 18", f"Cube {cube_p}/9 | Prism {prism_p}/9")

    col_orig, col_over = st.columns(2)
    with col_orig:
        st.markdown("**Original Scanned Sheet**")
        st.image(orig_bytes, use_container_width=True)
    with col_over:
        st.markdown("**OpenCV CAD Verification Overlay**")
        st.image(overlay_bytes, use_container_width=True)
        st.caption("Green = Dark edge matched | Amber = Faint edge | Red = Missing/misaligned | Orange dashed = Hidden-edge issue | Blue = Dimension lines")

    st.markdown("**Rubric Breakdown**")
    st.table(rubric["rows"])

    if vlm_dict:
        col_s, col_d = st.columns(2)
        with col_s:
            st.markdown("**Isometric Scale Audit (Gemini VLM)**")
            iso_a = vlm_dict["isometric_scale"]
            st.write(f"- **Present:** {'Yes' if iso_a['present'] else 'No'} | **30° & 45° Rays:** {'Yes' if iso_a['has_30_and_45_rays'] else 'No'}")
            st.write(f"- **Marked Lengths:** {', '.join(iso_a['requisite_lengths_marked']) or 'None detected'}")
            st.write(f"- **Notes:** {iso_a['notes']}")
        with col_d:
            st.markdown("**Dimensioning & Notation Audit (Gemini VLM)**")
            dim_a = vlm_dict["dimensioning_and_notation"]
            st.write(f"- **Observed Dimensions:** {', '.join(dim_a['dimensions_found']) or 'None detected'}")
            st.write(f"- **Missing Dimensions:** {', '.join(dim_a['missing_dimensions']) or 'None'}")
            st.write(f"- **Arrowheads & Conventions:** {dim_a['arrowheads_and_conventions']}")
            st.write(f"- **Notes:** {dim_a['notes']}")

        st.markdown("**Actionable Feedback**")
        for bullet in vlm_dict["combined_student_feedback"]:
            st.markdown(f"- {bullet}")
        st.caption(f"Line weights & neatness: {vlm_dict['neatness_and_line_weights']}")
    else:
        st.markdown("**Geometric Feedback (OpenCV)**")
        for bullet in cv_rep["feedback"]:
            st.markdown(f"- {bullet}")


def render_student_tab(settings: Dict[str, Any], dxf_path: Optional[str]) -> None:
    st.subheader("Submit Your Engineering Graphics Sheet")
    st.info(f"**Assignment Question:** {settings['problem_statement']}")

    if not dxf_path:
        st.error("No reference CAD key (`.dxf`) is configured yet. Ask faculty to upload one in the Teacher Dashboard.")
        return

    with st.form("student_submission_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        full_name = col1.text_input("Full Name *", placeholder="e.g., Arjun Menon")
        u_id = col2.text_input("U_ID / Register Number *", placeholder="e.g., RSET26ME042")
        email = st.text_input("College Email *", placeholder="e.g., arjun.me@rajagiritech.edu.in")
        uploaded_scan = st.file_uploader(
            "Upload Scanned Answer Sheet (.jpg, .jpeg, .png) *",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear full-page scan of your drawing sheet.",
        )
        submitted = st.form_submit_button("Submit & Grade Drawing", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not full_name.strip():
            errors.append("Full Name is required.")
        if not u_id.strip():
            errors.append("U_ID / Register Number is required.")
        if not email.strip() or "@" not in email:
            errors.append("A valid Email address is required.")
        if uploaded_scan is None:
            errors.append("Please upload your scanned drawing sheet.")

        if errors:
            for err in errors:
                st.error(err)
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scan_ext = Path(uploaded_scan.name).suffix or ".jpg"
            scan_path = str(tmp / f"student_scan{scan_ext}")
            orig_bytes = uploaded_scan.getvalue()
            Path(scan_path).write_bytes(orig_bytes)

            overlay_path = str(tmp / "graded_overlay.png")
            json_path = str(tmp / "grading_report.json")

            cv_kwargs: Dict[str, Any] = {
                "dxf_path": dxf_path,
                "scan_path": scan_path,
                "hidden_policy": settings["hidden_policy"],
                "tol_mm": float(settings["tol_mm"]),
                "out_overlay": overlay_path,
                "out_json": json_path,
            }
            cmode = settings["calib_mode"]
            cval = float(settings["calib_val"])
            if cmode == "Page Height (mm)":
                cv_kwargs["page_height_mm"] = cval
            elif cmode == "Page Width (mm)":
                cv_kwargs["page_width_mm"] = cval
            elif cmode == "Scanner DPI":
                cv_kwargs["dpi"] = cval
            else:
                cv_kwargs["unknown_dpi"] = True

            with st.spinner("Step 1/2: Aligning your scan to the CAD reference key & verifying geometry..."):
                try:
                    cv_report = grade_scan(**cv_kwargs)
                except Exception as exc:
                    st.error(f"Could not grade this scan automatically: {exc}")
                    return

            vlm_dict = None
            vlm_warning = None
            if settings.get("enable_vlm", True):
                api_override = st.session_state.get("session_api_key_override", "")
                with st.spinner("Step 2/2: Auditing Isometric Scale & dimensioning with Gemini VLM..."):
                    try:
                        vlm_dict = run_gemini_vlm_audit(
                            model_id=settings.get("model_id", "gemini-2.5-flash"),
                            problem_statement=settings["problem_statement"],
                            original_scan_path=scan_path,
                            overlay_path=overlay_path,
                            cv_report=cv_report,
                            api_key_override=api_override,
                        )
                    except Exception as exc:
                        vlm_warning = f"VLM notation audit unavailable ({exc}); graded using OpenCV geometry engine."

            overlay_bytes = Path(overlay_path).read_bytes()
            rubric = compute_hybrid_rubric(cv_report, vlm_dict)

            sub_id = save_submission(
                full_name=full_name,
                u_id=u_id,
                email=email,
                scan_filename=uploaded_scan.name,
                cv_report=cv_report,
                vlm_dict=vlm_dict,
                rubric=rubric,
                orig_bytes=orig_bytes,
                overlay_bytes=overlay_bytes,
            )
            gsheet_status = maybe_sync_to_google_sheet(sub_id)

            st.session_state["last_student_submission"] = {
                "sub_id": sub_id,
                "full_name": full_name.strip(),
                "u_id": u_id.strip(),
                "orig_bytes": orig_bytes,
                "overlay_bytes": overlay_bytes,
                "cv_report": cv_report,
                "vlm_dict": vlm_dict,
                "rubric": rubric,
                "vlm_warning": vlm_warning,
                "gsheet_status": gsheet_status,
            }

    if "last_student_submission" in st.session_state:
        res = st.session_state["last_student_submission"]
        st.success(
            f"Submission #{res['sub_id']} recorded for **{res['full_name']}** (`{res['u_id']}`)."
            + (f" ({res['gsheet_status']})" if res.get("gsheet_status") else "")
        )
        if res.get("vlm_warning"):
            st.warning(res["vlm_warning"])

        if settings.get("show_results_to_students", True):
            st.divider()
            render_result_details(
                orig_bytes=res["orig_bytes"],
                overlay_bytes=res["overlay_bytes"],
                cv_rep=res["cv_report"],
                vlm_dict=res["vlm_dict"],
                rubric=res["rubric"],
            )
            st.download_button(
                "Download Your Annotated Feedback Sheet (PNG)",
                data=res["overlay_bytes"],
                file_name=f"feedback_{res['u_id']}.png",
                mime="image/png",
            )
        else:
            st.info("Your sheet has been submitted for faculty review. Marks will be published by your instructor.")


def render_teacher_tab(settings: Dict[str, Any], dxf_path: Optional[str]) -> None:
    st.subheader("Teacher Dashboard & Rubric Gradebook")

    expected_pwd = get_secret("TEACHER_PASSWORD", "faculty123")
    using_default_pwd = not bool(get_secret("TEACHER_PASSWORD"))

    if not st.session_state.get("teacher_authenticated", False):
        with st.form("teacher_login_form"):
            pwd_input = st.text_input("Enter Teacher Password", type="password")
            login_btn = st.form_submit_button("Unlock Teacher Dashboard", type="primary")
        if using_default_pwd:
            st.caption("Local dev default password is `faculty123`. Set `TEACHER_PASSWORD` in Streamlit Secrets before sharing publicly.")
        if login_btn:
            if pwd_input == expected_pwd:
                st.session_state["teacher_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    # Logged-in header bar
    top_l, top_r = st.columns([5, 1])
    if using_default_pwd:
        top_l.warning("Using default password (`faculty123`). Set `TEACHER_PASSWORD` in Streamlit Secrets for production.")
    else:
        top_l.success("Authenticated as Faculty.")
    if top_r.button("Lock / Logout", use_container_width=True):
        st.session_state["teacher_authenticated"] = False
        st.rerun()

    # -----------------------------------------------------------------------
    # 1. Tabulated Gradebook & Export
    # -----------------------------------------------------------------------
    df = fetch_gradebook_df()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Submissions", len(df))
    if not df.empty:
        k2.metric("Class Average", f"{df['Approved Marks'].mean():.2f} / 10")
        k3.metric("Highest Score", f"{df['Approved Marks'].max():.2f} / 10")
        view_errs = int((df["Detected Scale"] == "View (1.0)").sum())
        k4.metric("1:1 View Scale Errors", view_errs)
    else:
        k2.metric("Class Average", "—")
        k3.metric("Highest Score", "—")
        k4.metric("1:1 View Scale Errors", 0)

    st.markdown("### Tabulated Student Gradebook (with Rubric Division)")
    if df.empty:
        st.info("No student submissions yet. Submit a test scan in the **Student Submission** tab to populate the gradebook.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        xlsx_bytes = build_excel_bytes(df)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Export Gradebook as CSV (.csv)",
            data=csv_bytes,
            file_name=f"eg_gradebook_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl2.download_button(
            "Export Gradebook as Excel (.xlsx)",
            data=xlsx_bytes,
            file_name=f"eg_gradebook_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # -------------------------------------------------------------------
        # 2. Individual Student Inspection & Faculty Override
        # -------------------------------------------------------------------
        st.divider()
        st.markdown("### Inspect Student Sheet & Faculty Override")
        options = {
            f"#{int(r['ID'])} | {r['U_ID']} - {r['Full Name']} ({r['Approved Marks']:.2f}/{r['Max Marks']:.0f})": int(r["ID"])
            for _, r in df.iterrows()
        }
        selected_label = st.selectbox("Select Student Submission", list(options.keys()))
        sub_id = options[selected_label]
        detail = fetch_submission_detail(sub_id)

        if detail:
            st.caption(
                f"**Email:** {detail['email']}  |  **File:** `{detail['scan_filename']}`  |  "
                f"**Submitted:** {detail['submitted_at']}  |  **Mode:** {detail['grading_mode']}"
            )
            render_result_details(
                orig_bytes=detail["orig_image_blob"],
                overlay_bytes=detail["overlay_image_blob"],
                cv_rep=detail["cv_report"],
                vlm_dict=detail["vlm_audit"],
                rubric=detail["rubric"],
                approved_marks=detail["approved_marks"],
            )

            with st.form(f"override_form_{sub_id}"):
                oc1, oc2 = st.columns([1, 3])
                new_marks = oc1.number_input(
                    "Approved Marks",
                    min_value=0.0,
                    max_value=float(detail["max_marks"]),
                    value=float(detail["approved_marks"]),
                    step=0.25,
                )
                new_remark = oc2.text_input("Faculty Remark", value=detail["faculty_remark"] or "")
                save_ov = st.form_submit_button("Save Faculty Override", type="primary")
            if save_ov:
                update_faculty_override(sub_id, new_marks, new_remark)
                st.success(f"Updated Submission #{sub_id} to {new_marks:.2f} marks.")
                st.rerun()

            ac1, ac2, ac3 = st.columns(3)
            ac1.download_button(
                "Download Overlay (PNG)",
                data=detail["overlay_image_blob"],
                file_name=f"overlay_{detail['u_id']}_{sub_id}.png",
                mime="image/png",
                use_container_width=True,
            )
            ac2.download_button(
                "Download Full JSON",
                data=json.dumps({
                    "id": detail["id"],
                    "u_id": detail["u_id"],
                    "full_name": detail["full_name"],
                    "email": detail["email"],
                    "approved_marks": detail["approved_marks"],
                    "faculty_remark": detail["faculty_remark"],
                    "rubric": detail["rubric"],
                    "cv_report": detail["cv_report"],
                    "vlm_audit": detail["vlm_audit"],
                }, indent=2),
                file_name=f"report_{detail['u_id']}_{sub_id}.json",
                mime="application/json",
                use_container_width=True,
            )
            if ac3.button("Delete This Submission", use_container_width=True):
                delete_submission(sub_id)
                st.warning(f"Deleted Submission #{sub_id}.")
                st.rerun()

    # -----------------------------------------------------------------------
    # 3. Grading Configuration & Reference Key (Teacher-Only)
    # -----------------------------------------------------------------------
    st.divider()
    with st.expander("Assignment & Grading Engine Settings", expanded=False):
        st.caption(f"**Active Reference DXF:** `{dxf_path or 'None found'}`")
        new_dxf = st.file_uploader("Upload / Replace Reference CAD Key (.dxf)", type=["dxf"], key="teacher_dxf_up")
        if new_dxf is not None:
            ACTIVE_DXF_PATH.write_bytes(new_dxf.getvalue())
            st.success(f"Saved `{new_dxf.name}` as active reference key (`{ACTIVE_DXF_PATH.name}`).")

        env_key_found = bool(get_secret("GEMINI_API_KEY"))
        if env_key_found:
            st.success("`GEMINI_API_KEY` detected in Streamlit Secrets / Environment.")
        else:
            st.warning("`GEMINI_API_KEY` not detected. Add it to Streamlit Secrets or enter a temporary session key below.")
            st.session_state["session_api_key_override"] = st.text_input(
                "Temporary Session Gemini API Key",
                value=st.session_state.get("session_api_key_override", ""),
                type="password",
            )

        with st.form("teacher_settings_form"):
            p_stmt = st.text_area("Problem Statement (shown to students)", value=settings["problem_statement"], height=70)
            sc1, sc2 = st.columns(2)
            calib_modes = ["Page Height (mm)", "Page Width (mm)", "Scanner DPI", "Unknown DPI (Proportions Only)"]
            c_idx = calib_modes.index(settings["calib_mode"]) if settings["calib_mode"] in calib_modes else 0
            c_mode = sc1.selectbox("Physical Scale Calibration", calib_modes, index=c_idx)
            c_val = sc2.number_input("Calibration Value (mm or DPI)", min_value=0.0, max_value=600.0, value=float(settings["calib_val"]), step=1.0)

            sc3, sc4 = st.columns(2)
            t_mm = sc3.slider("Geometric Tolerance (mm)", min_value=1.0, max_value=5.0, value=float(settings["tol_mm"]), step=0.25)
            h_pol = sc4.selectbox("Hidden Edges Policy", ["omit", "require_dashed"], index=0 if settings["hidden_policy"] == "omit" else 1)

            sc5, sc6 = st.columns(2)
            en_vlm = sc5.checkbox("Enable Gemini VLM Audit", value=bool(settings["enable_vlm"]))
            show_stu = sc6.checkbox("Show instant score & overlay to students", value=bool(settings["show_results_to_students"]))

            models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
            cur_model = settings.get("model_id", "gemini-2.5-flash")
            if cur_model not in models:
                models.append(cur_model)
            m_id = st.selectbox("Gemini Model", models, index=models.index(cur_model))

            if st.form_submit_button("Save Grading Settings", type="primary"):
                save_settings({
                    "problem_statement": p_stmt.strip(),
                    "calib_mode": c_mode,
                    "calib_val": float(c_val),
                    "tol_mm": float(t_mm),
                    "hidden_policy": h_pol,
                    "enable_vlm": bool(en_vlm),
                    "model_id": m_id,
                    "show_results_to_students": bool(show_stu),
                })
                st.success("Grading settings saved for all future submissions.")
                st.rerun()


def render_streamlit_app() -> None:
    st.set_page_config(page_title="EG Auto-Grader Portal", layout="wide")
    init_db()
    settings = load_settings()
    dxf_path = find_reference_dxf()

    st.title("Engineering Graphics Auto-Grader Portal")
    st.caption("Hybrid CAD-DXF + OpenCV geometric verification & Gemini VLM notation audit")

    tab_student, tab_teacher = st.tabs(["Student Submission", "Teacher Dashboard"])
    with tab_student:
        render_student_tab(settings, dxf_path)
    with tab_teacher:
        render_teacher_tab(settings, dxf_path)


def _is_running_in_streamlit() -> bool:
    if not HAS_STREAMLIT:
        return False
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Borderless DXF + OpenCV Isometric Projection Grader (CLI Mode)")
    parser.add_argument("--dxf", required=True, help="Path to isometric_prism_on_cube-v3.dxf")
    parser.add_argument("--scan", required=True, help="Path to student scan image (PNG/JPG)")
    parser.add_argument("--dpi", type=float, default=None, help="Known scanner DPI")
    parser.add_argument("--page-width-mm", type=float, default=None, help="Known physical page width in mm")
    parser.add_argument("--page-height-mm", type=float, default=290.0, help="Known physical page height in mm")
    parser.add_argument("--unknown-dpi", action="store_true", help="Scale-free proportional grading")
    parser.add_argument("--hidden-policy", choices=["omit", "require_dashed"], default="omit")
    parser.add_argument("--tol-mm", type=float, default=2.5)
    parser.add_argument("--out-overlay", default="graded_overlay.png")
    parser.add_argument("--out-json", default="grading_report.json")
    args = parser.parse_args()

    rep = grade_scan(
        dxf_path=args.dxf, scan_path=args.scan,
        dpi=args.dpi, page_width_mm=args.page_width_mm,
        page_height_mm=None if (args.dpi or args.page_width_mm or args.unknown_dpi) else args.page_height_mm,
        unknown_dpi=args.unknown_dpi, hidden_policy=args.hidden_policy,
        tol_mm=args.tol_mm, out_overlay=args.out_overlay, out_json=args.out_json,
    )
    print(json.dumps({
        "scale_verdict": rep["scale_detection"]["verdict"],
        "deskew_deg": rep["calibration"]["deskew_angle_deg"],
        "rubric_marks": rep["rubric_marks"],
        "total_marks": f"{rep['total_marks']}/{rep['max_marks']}",
        "feedback": rep["feedback"],
    }, indent=2))


if _is_running_in_streamlit():
    render_streamlit_app()
elif __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Clicking VS Code's Play button launches the Streamlit web app automatically
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
