
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import folium
from folium.raster_layers import ImageOverlay
from folium.features import DivIcon
from folium.plugins import Draw
from streamlit_folium import st_folium
from pyproj import Transformer
from matplotlib import colors, colormaps
from matplotlib.path import Path as MplPath
import plotly.graph_objects as go

st.set_page_config(page_title="Bozuyuk InSAR Dashboard", layout="wide")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
AOI_CENTER_LAT = 39.86891807427488
AOI_CENTER_LON = 30.17805156594522
AOI_RADIUS_M = 5000
SRC_EPSG = "EPSG:32635"

BASE_DIR = Path(__file__).resolve().parent
MINTPY_DIR = BASE_DIR / "data"
VELOCITY_H5 = MINTPY_DIR / "velocity.h5"
COH_H5 = MINTPY_DIR / "temporalCoherence.h5"
TS_H5 = MINTPY_DIR / "timeseries.h5"

st.title("Bozuyuk InSAR Dashboard")
st.caption("Selected-interval cumulative displacement, interval velocity, point analysis, A-B profile and polygon statistics")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "point_a" not in st.session_state:
    st.session_state.point_a = None
if "point_b" not in st.session_state:
    st.session_state.point_b = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [AOI_CENTER_LAT, AOI_CENTER_LON]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 12

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------
missing = [p.name for p in [VELOCITY_H5, COH_H5, TS_H5] if not p.exists()]
if missing:
    st.error(f"Eksik dosyalar: {', '.join(missing)}")
    st.stop()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def _to_text(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return str(v)

@st.cache_data
def load_h5_dataset(file_path: str, dataset_name: str):
    with h5py.File(file_path, "r") as f:
        arr = f[dataset_name][:].astype("float32")
        attrs = {k: _to_text(v) for k, v in f.attrs.items()}
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr, attrs

@st.cache_data
def load_dates(file_path: str):
    with h5py.File(file_path, "r") as f:
        raw_dates = f["date"][:]
    return pd.to_datetime([d.decode("utf-8") for d in raw_dates], format="%Y%m%d")

@st.cache_data
def load_timeseries_slice(file_path: str, idx: int):
    with h5py.File(file_path, "r") as f:
        arr = f["timeseries"][idx, :, :].astype("float32")
    return np.where(np.isfinite(arr), arr, np.nan)

@st.cache_data
def load_point_timeseries(file_path: str, row: int, col: int):
    with h5py.File(file_path, "r") as f:
        arr = f["timeseries"][:, row, col].astype("float32")
    return np.where(np.isfinite(arr), arr, np.nan)

def get_projected_axes(attrs, shape):
    x_first = float(attrs["X_FIRST"])
    y_first = float(attrs["Y_FIRST"])
    x_step = float(attrs["X_STEP"])
    y_step = float(attrs["Y_STEP"])
    h, w = shape
    xs = x_first + np.arange(w) * x_step
    ys = y_first + np.arange(h) * y_step
    return xs, ys, x_first, y_first, x_step, y_step

def get_projected_bounds(attrs, shape):
    xs, ys, x_first, y_first, x_step, y_step = get_projected_axes(attrs, shape)
    h, w = shape
    x_last = x_first + (w - 1) * x_step
    y_last = y_first + (h - 1) * y_step
    west = min(x_first, x_last) - abs(x_step) / 2.0
    east = max(x_first, x_last) + abs(x_step) / 2.0
    south = min(y_first, y_last) - abs(y_step) / 2.0
    north = max(y_first, y_last) + abs(y_step) / 2.0
    return west, south, east, north

def projected_bounds_to_latlon(bounds_proj, src_epsg):
    west, south, east, north = bounds_proj
    t = Transformer.from_crs(src_epsg, "EPSG:4326", always_xy=True)
    sw_lon, sw_lat = t.transform(west, south)
    ne_lon, ne_lat = t.transform(east, north)
    return [[sw_lat, sw_lon], [ne_lat, ne_lon]]

def nearest_pixel_from_latlon(lat, lon, xs_proj, ys_proj, src_epsg):
    t = Transformer.from_crs("EPSG:4326", src_epsg, always_xy=True)
    x_proj, y_proj = t.transform(lon, lat)
    col = int(np.argmin(np.abs(xs_proj - x_proj)))
    row = int(np.argmin(np.abs(ys_proj - y_proj)))
    return row, col, x_proj, y_proj

def array_to_rgba(arr, cmap_name, vmin, vmax):
    arr2 = np.array(arr, dtype="float32")
    valid = np.isfinite(arr2)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = colormaps.get_cmap(cmap_name)
    rgba = cmap(norm(np.nan_to_num(arr2, nan=vmin)))
    rgba[..., 3] = np.where(valid, 1.0, 0.0)
    return (rgba * 255).astype("uint8")

def nearest_date_indices(date_index, start_date, end_date):
    dvals = date_index.values.astype("datetime64[D]")
    start64 = np.datetime64(start_date)
    end64 = np.datetime64(end_date)
    i0 = int(np.abs(dvals - start64).argmin())
    i1 = int(np.abs(dvals - end64).argmin())
    if i0 > i1:
        i0, i1 = i1, i0
    return i0, i1

def safe_minmax(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(valid)), float(np.nanmax(valid))

def build_histogram(values, title, x_title):
    valid = values[np.isfinite(values)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=valid, nbinsx=60))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def build_point_info(lat, lon, xs_proj, ys_proj, src_epsg, shape):
    row, col, x_proj, y_proj = nearest_pixel_from_latlon(lat, lon, xs_proj, ys_proj, src_epsg)
    row = max(0, min(row, shape[0] - 1))
    col = max(0, min(col, shape[1] - 1))
    return {
        "lat": lat,
        "lon": lon,
        "row": row,
        "col": col,
        "x_proj": float(x_proj),
        "y_proj": float(y_proj),
    }

def sample_line_profile(point_a, point_b, arrays_dict, n_samples=None):
    row0, col0 = point_a["row"], point_a["col"]
    row1, col1 = point_b["row"], point_b["col"]
    x0, y0 = point_a["x_proj"], point_a["y_proj"]
    x1, y1 = point_b["x_proj"], point_b["y_proj"]

    if n_samples is None:
        n_samples = int(max(abs(row1 - row0), abs(col1 - col0))) + 1
        n_samples = max(n_samples, 25)

    rows = np.round(np.linspace(row0, row1, n_samples)).astype(int)
    cols = np.round(np.linspace(col0, col1, n_samples)).astype(int)

    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    rows = rows[keep]
    cols = cols[keep]

    x_line = np.linspace(x0, x1, len(rows))
    y_line = np.linspace(y0, y1, len(rows))
    dist_m = np.sqrt((x_line - x0) ** 2 + (y_line - y0) ** 2)

    out = {
        "distance_m": dist_m,
        "row": rows,
        "col": cols,
        "x_proj": x_line,
        "y_proj": y_line,
    }
    for name, arr in arrays_dict.items():
        out[name] = arr[rows, cols]
    return out

def add_labeled_marker(m, lat, lon, label, color):
    folium.Marker(
        location=[lat, lon],
        icon=DivIcon(
            icon_size=(26, 26),
            icon_anchor=(13, 13),
            html=f'<div style="font-size:18px;font-weight:bold;color:{color};background:white;border:2px solid {color};border-radius:50%;width:26px;height:26px;line-height:22px;text-align:center;">{label}</div>'
        ),
        tooltip=f"Point {label}"
    ).add_to(m)

def build_legend_html(overlay_name, overlay_unit, scale_min, scale_max):
    tick_values = np.linspace(scale_min, scale_max, 7)
    cmap = colormaps.get_cmap("RdBu_r")
    hex_colors = [colors.to_hex(cmap(v)) for v in np.linspace(0, 1, 7)]
    gradient = ", ".join([f"{hex_colors[i]} {int(i * 100 / 6)}%" for i in range(7)])
    tick_html = "".join([f'<div style="flex:1;text-align:center;">{v:.1f}</div>' for v in tick_values])

    return f"""
    <div style="
        background:white;
        border:1px solid #d0d0d0;
        border-radius:10px;
        padding:12px 14px;
        margin-top:6px;
        margin-bottom:10px;
        box-shadow:0 1px 4px rgba(0,0,0,0.08);
        font-family:Arial, sans-serif;
    ">
        <div style="font-weight:700; margin-bottom:8px;">Legend — {overlay_name}</div>

        <div style="
            height:18px;
            border-radius:8px;
            background: linear-gradient(to right, {gradient});
            border:1px solid #bdbdbd;
        "></div>

        <div style="display:flex; font-size:11px; margin-top:6px;">
            {tick_html}
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            font-size:12px;
            color:#444;
            margin-top:8px;
            font-weight:600;
            gap:10px;
        ">
            <span>Sol uç = düşük / daha negatif</span>
            <span>Orta = 0’a yakın</span>
            <span>Sağ uç = yüksek / daha pozitif</span>
        </div>

        <div style="font-size:12px; color:#666; margin-top:6px;">
            Unit: {overlay_unit}
        </div>
    </div>
    """

def get_polygon_feature(map_state):
    if not map_state:
        return None

    last_active = map_state.get("last_active_drawing")
    if isinstance(last_active, dict):
        geom = last_active.get("geometry", {})
        if geom.get("type") == "Polygon":
            return last_active

    all_drawings = map_state.get("all_drawings")
    if isinstance(all_drawings, list):
        for feat in reversed(all_drawings):
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                return feat
    return None

def polygon_feature_to_mask(feature, xs_proj, ys_proj, src_epsg):
    if feature is None:
        return None

    geom = feature.get("geometry", {})
    if geom.get("type") != "Polygon":
        return None

    coords = geom.get("coordinates", [])
    if not coords or not coords[0]:
        return None

    outer_ring_lonlat = coords[0]
    transformer = Transformer.from_crs("EPSG:4326", src_epsg, always_xy=True)
    poly_xy = np.array([transformer.transform(lon, lat) for lon, lat in outer_ring_lonlat], dtype=float)

    xx, yy = np.meshgrid(xs_proj, ys_proj)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    path = MplPath(poly_xy)
    return path.contains_points(pts).reshape(xx.shape)

def compute_polygon_mean_timeseries(file_path, idx_start, idx_end, rows, cols):
    values = []
    for idx in range(idx_start, idx_end + 1):
        sl = load_timeseries_slice(file_path, idx) * 1000.0
        vals = sl[rows, cols]
        values.append(np.nanmean(vals) if vals.size else np.nan)
    return np.array(values, dtype=float)

# --------------------------------------------------
# LOAD STATIC DATA
# --------------------------------------------------
velocity_m_yr, vel_attrs = load_h5_dataset(str(VELOCITY_H5), "velocity")
velocity_std_m_yr, _ = load_h5_dataset(str(VELOCITY_H5), "velocityStd")
temporal_coh, _ = load_h5_dataset(str(COH_H5), "temporalCoherence")
dates = load_dates(str(TS_H5))

velocity_std_mm_yr = velocity_std_m_yr * 1000.0
xs_proj, ys_proj, *_ = get_projected_axes(vel_attrs, velocity_m_yr.shape)
bounds_proj = get_projected_bounds(vel_attrs, velocity_m_yr.shape)
bounds_latlon = projected_bounds_to_latlon(bounds_proj, SRC_EPSG)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Coverage")
st.sidebar.write(f"Data range: **{dates.min().date()} → {dates.max().date()}**")
st.sidebar.write(f"Epoch count: **{len(dates)}**")

st.sidebar.header("Selected time interval")
selected_range = st.sidebar.slider(
    "Time range",
    min_value=dates.min().date(),
    max_value=dates.max().date(),
    value=(dates.min().date(), dates.max().date()),
    format="YYYY-MM-DD"
)

overlay_name = st.sidebar.radio(
    "Map overlay",
    ["Interval velocity (mm/year)", "Cumulative displacement (mm)"],
    index=0
)

opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.65, 0.05)

st.sidebar.header("Quality filters")
coh_thr = st.sidebar.slider("Min temporal coherence", 0.00, 1.00, 0.70, 0.01)
std_thr = st.sidebar.number_input("Max velocityStd (mm/year)", min_value=0.0, value=10.0, step=0.5)

st.sidebar.header("Display scale")
vel_vmin = st.sidebar.number_input("Velocity min (mm/year)", value=-60.0, step=5.0)
vel_vmax = st.sidebar.number_input("Velocity max (mm/year)", value=60.0, step=5.0)
disp_vmin = st.sidebar.number_input("Displacement min (mm)", value=-100.0, step=5.0)
disp_vmax = st.sidebar.number_input("Displacement max (mm)", value=100.0, step=5.0)

# --------------------------------------------------
# DATE RANGE PRODUCTS
# --------------------------------------------------
req_start_date, req_end_date = selected_range
idx_start, idx_end = nearest_date_indices(dates, req_start_date, req_end_date)

eff_start = dates[idx_start]
eff_end = dates[idx_end]
duration_days = int((eff_end - eff_start).days)

if duration_days <= 0:
    st.warning("Seçilen aralık en az iki farklı tarihe yayılmalı.")
    st.stop()

ts_start_m = load_timeseries_slice(str(TS_H5), idx_start)
ts_end_m = load_timeseries_slice(str(TS_H5), idx_end)

cumulative_disp_mm = (ts_end_m - ts_start_m) * 1000.0
interval_vel_mm_yr = cumulative_disp_mm / duration_days * 365.25

quality_mask = (
    np.isfinite(temporal_coh)
    & np.isfinite(velocity_std_mm_yr)
    & (temporal_coh >= coh_thr)
    & (velocity_std_mm_yr <= std_thr)
)

cumulative_disp_mm = np.where(quality_mask, cumulative_disp_mm, np.nan)
interval_vel_mm_yr = np.where(quality_mask, interval_vel_mm_yr, np.nan)

if overlay_name == "Interval velocity (mm/year)":
    overlay_arr = interval_vel_mm_yr
    overlay_rgba = array_to_rgba(overlay_arr, "RdBu_r", vel_vmin, vel_vmax)
    overlay_unit = "mm/year"
    legend_min, legend_max = vel_vmin, vel_vmax
else:
    overlay_arr = cumulative_disp_mm
    overlay_rgba = array_to_rgba(overlay_arr, "RdBu_r", disp_vmin, disp_vmax)
    overlay_unit = "mm"
    legend_min, legend_max = disp_vmin, disp_vmax

overlay_min, overlay_max = safe_minmax(overlay_arr)
coh_valid = temporal_coh[quality_mask]
filtered_count = int(np.isfinite(overlay_arr).sum())

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
st.info(
    f"Requested range: {req_start_date} → {req_end_date} | "
    f"Effective epochs: {eff_start.date()} → {eff_end.date()} | "
    f"Duration: {duration_days} days"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Overlay min", f"{overlay_min:.2f} {overlay_unit}" if np.isfinite(overlay_min) else "NA")
m2.metric("Overlay max", f"{overlay_max:.2f} {overlay_unit}" if np.isfinite(overlay_max) else "NA")
m3.metric("Mean coherence", f"{np.nanmean(coh_valid):.3f}" if coh_valid.size else "NA")
m4.metric("Filtered pixels", f"{filtered_count:,}")

# --------------------------------------------------
# MAP
# --------------------------------------------------
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles="CartoDB positron",
    control_scale=True
)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
    overlay=False,
    control=True
).add_to(m)

ImageOverlay(
    image=overlay_rgba,
    bounds=bounds_latlon,
    opacity=opacity,
    interactive=True,
    cross_origin=False,
    origin="upper",
    mercator_project=True,
    name=overlay_name
).add_to(m)

folium.Marker(
    location=[AOI_CENTER_LAT, AOI_CENTER_LON],
    icon=DivIcon(
        icon_size=(24, 24),
        icon_anchor=(12, 12),
        html='<div style="font-size:22px;color:gold;text-shadow:0 0 3px black;">★</div>'
    ),
    tooltip="AOI center"
).add_to(m)

folium.Circle(
    location=[AOI_CENTER_LAT, AOI_CENTER_LON],
    radius=AOI_RADIUS_M,
    color="gold",
    weight=2,
    fill=False,
    tooltip="AOI radius = 5 km"
).add_to(m)

if st.session_state.point_a is not None:
    add_labeled_marker(m, st.session_state.point_a["lat"], st.session_state.point_a["lon"], "A", "#d62728")

if st.session_state.point_b is not None:
    add_labeled_marker(m, st.session_state.point_b["lat"], st.session_state.point_b["lon"], "B", "#1f77b4")

if st.session_state.point_a is not None and st.session_state.point_b is not None:
    folium.PolyLine(
        locations=[
            [st.session_state.point_a["lat"], st.session_state.point_a["lon"]],
            [st.session_state.point_b["lat"], st.session_state.point_b["lon"]],
        ],
        color="black",
        weight=3,
        opacity=0.9,
        tooltip="A-B profile line"
    ).add_to(m)

Draw(
    export=False,
    draw_options={
        "polyline": False,
        "polygon": True,
        "rectangle": True,
        "circle": False,
        "marker": False,
        "circlemarker": False
    },
    edit_options={"edit": True, "remove": True}
).add_to(m)

folium.LayerControl().add_to(m)

map_state = st_folium(
    m,
    key="main_map",
    height=720,
    width=None,
    center=tuple(st.session_state.map_center),
    zoom=st.session_state.map_zoom,
    returned_objects=["last_clicked", "zoom", "center", "bounds", "all_drawings", "last_active_drawing"]
)

if map_state:
    if map_state.get("zoom") is not None:
        st.session_state.map_zoom = map_state["zoom"]

    center = map_state.get("center")
    if isinstance(center, dict) and ("lat" in center) and ("lng" in center):
        st.session_state.map_center = [center["lat"], center["lng"]]

# --------------------------------------------------
# LEGEND
# --------------------------------------------------
components.html(
    build_legend_html(overlay_name, overlay_unit, legend_min, legend_max),
    height=150,
    scrolling=False
)

# --------------------------------------------------
# MAP INTERACTION PANEL
# --------------------------------------------------
clicked = map_state.get("last_clicked", None) if map_state else None
polygon_feature = get_polygon_feature(map_state)

st.subheader("Map interaction")

if clicked:
    st.write(f"Last clicked point: **{clicked['lat']:.6f}, {clicked['lng']:.6f}**")
else:
    st.write("Last clicked point: —")

if polygon_feature is not None:
    st.write("Active polygon: **available**")
else:
    st.write("Active polygon: —")

cbtn1, cbtn2, cbtn3 = st.columns(3)

if cbtn1.button("Use last clicked as Point A", use_container_width=True):
    if clicked:
        st.session_state.point_a = build_point_info(
            clicked["lat"], clicked["lng"], xs_proj, ys_proj, SRC_EPSG, overlay_arr.shape
        )
        st.rerun()

if cbtn2.button("Use last clicked as Point B", use_container_width=True):
    if clicked:
        st.session_state.point_b = build_point_info(
            clicked["lat"], clicked["lng"], xs_proj, ys_proj, SRC_EPSG, overlay_arr.shape
        )
        st.rerun()

if cbtn3.button("Clear A / B", use_container_width=True):
    st.session_state.point_a = None
    st.session_state.point_b = None
    st.rerun()

# --------------------------------------------------
# SUMMARY HISTOGRAMS
# --------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        build_histogram(overlay_arr, f"{overlay_name} distribution", overlay_unit),
        width="stretch"
    )
with c2:
    st.plotly_chart(
        build_histogram(coh_valid, "Temporal coherence distribution", "coherence"),
        width="stretch"
    )

# --------------------------------------------------
# CLICKED POINT ANALYSIS
# --------------------------------------------------
if clicked:
    row, col, x_proj, y_proj = nearest_pixel_from_latlon(
        clicked["lat"], clicked["lng"], xs_proj, ys_proj, SRC_EPSG
    )

    row = max(0, min(row, overlay_arr.shape[0] - 1))
    col = max(0, min(col, overlay_arr.shape[1] - 1))

    point_ts_m = load_point_timeseries(str(TS_H5), row, col)
    point_ts_mm = point_ts_m * 1000.0

    point_sel_dates = dates[idx_start:idx_end + 1]
    point_sel_abs_mm = point_ts_mm[idx_start:idx_end + 1]
    point_sel_cum_mm = point_sel_abs_mm - point_sel_abs_mm[0]

    dt_days = np.diff(point_sel_dates.values.astype("datetime64[D]")).astype(int).astype(float)
    if len(point_sel_cum_mm) >= 2:
        point_rate_mm_yr = np.diff(point_sel_cum_mm) / dt_days * 365.25
        point_rate_dates = point_sel_dates[1:]
    else:
        point_rate_mm_yr = np.array([])
        point_rate_dates = []

    point_total_disp = float(point_sel_cum_mm[-1]) if len(point_sel_cum_mm) else np.nan
    point_avg_vel = point_total_disp / duration_days * 365.25 if np.isfinite(point_total_disp) else np.nan
    point_coh = float(temporal_coh[row, col]) if np.isfinite(temporal_coh[row, col]) else np.nan
    point_std = float(velocity_std_mm_yr[row, col]) if np.isfinite(velocity_std_mm_yr[row, col]) else np.nan

    st.subheader("Selected point analysis")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Clicked lat/lon", f"{clicked['lat']:.5f}, {clicked['lng']:.5f}")
    a2.metric("Row/col", f"{row}, {col}")
    a3.metric("Point coherence", f"{point_coh:.3f}" if np.isfinite(point_coh) else "NA")
    a4.metric("Point velocityStd", f"{point_std:.2f} mm/y" if np.isfinite(point_std) else "NA")

    b1, b2 = st.columns(2)
    b1.metric("Point total displacement", f"{point_total_disp:.2f} mm" if np.isfinite(point_total_disp) else "NA")
    b2.metric("Point average velocity", f"{point_avg_vel:.2f} mm/y" if np.isfinite(point_avg_vel) else "NA")

    d1, d2 = st.columns(2)
    with d1:
        fig_disp = go.Figure()
        fig_disp.add_trace(go.Scatter(x=point_sel_dates, y=point_sel_cum_mm, mode="lines+markers", name="Cumulative displacement"))
        fig_disp.update_layout(
            title="Cumulative displacement vs date",
            xaxis_title="Date",
            yaxis_title="Cumulative displacement (mm)",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_disp, width="stretch")

    with d2:
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(x=point_rate_dates, y=point_rate_mm_yr, mode="lines+markers", name="Rate"))
        fig_rate.update_layout(
            title="Deformation rate vs date",
            xaxis_title="Date",
            yaxis_title="Rate (mm/year)",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_rate, width="stretch")

# --------------------------------------------------
# A-B PROFILE
# --------------------------------------------------
if st.session_state.point_a is not None and st.session_state.point_b is not None:
    profile = sample_line_profile(
        st.session_state.point_a,
        st.session_state.point_b,
        {
            "velocity_mm_yr": interval_vel_mm_yr,
            "displacement_mm": cumulative_disp_mm,
            "coherence": temporal_coh,
        }
    )

    dist_km = profile["distance_m"] / 1000.0

    st.subheader("A-B line profile")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("A point", f"{st.session_state.point_a['lat']:.4f}, {st.session_state.point_a['lon']:.4f}")
    p2.metric("B point", f"{st.session_state.point_b['lat']:.4f}, {st.session_state.point_b['lon']:.4f}")
    p3.metric("Profile length", f"{dist_km[-1]:.2f} km")
    p4.metric("Sample count", f"{len(dist_km)}")

    fig_prof_vel = go.Figure()
    fig_prof_vel.add_trace(go.Scatter(x=dist_km, y=profile["velocity_mm_yr"], mode="lines+markers", name="Velocity"))
    fig_prof_vel.update_layout(
        title="Interval velocity along A-B line",
        xaxis_title="Distance along line (km)",
        yaxis_title="Velocity (mm/year)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig_prof_disp = go.Figure()
    fig_prof_disp.add_trace(go.Scatter(x=dist_km, y=profile["displacement_mm"], mode="lines+markers", name="Cumulative displacement"))
    fig_prof_disp.update_layout(
        title="Cumulative displacement along A-B line",
        xaxis_title="Distance along line (km)",
        yaxis_title="Cumulative displacement (mm)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig_prof_coh = go.Figure()
    fig_prof_coh.add_trace(go.Scatter(x=dist_km, y=profile["coherence"], mode="lines+markers", name="Coherence"))
    fig_prof_coh.update_layout(
        title="Temporal coherence along A-B line",
        xaxis_title="Distance along line (km)",
        yaxis_title="Coherence",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(fig_prof_vel, width="stretch")
    with r2:
        st.plotly_chart(fig_prof_disp, width="stretch")
    st.plotly_chart(fig_prof_coh, width="stretch")

# --------------------------------------------------
# POLYGON ANALYSIS
# --------------------------------------------------
if polygon_feature is not None:
    polygon_mask_raw = polygon_feature_to_mask(polygon_feature, xs_proj, ys_proj, SRC_EPSG)

    st.subheader("Polygon analysis")

    if polygon_mask_raw is None:
        st.warning("Polygon okunamadı.")
    else:
        valid_poly_mask = polygon_mask_raw & np.isfinite(overlay_arr)
        poly_rows, poly_cols = np.where(valid_poly_mask)

        if len(poly_rows) == 0:
            st.warning("Polygon içinde geçerli piksel bulunamadı.")
        else:
            poly_overlay_vals = overlay_arr[valid_poly_mask]
            poly_coh_vals = temporal_coh[polygon_mask_raw & np.isfinite(temporal_coh)]

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Polygon pixels", f"{len(poly_rows):,}")
            pm2.metric("Polygon mean", f"{np.nanmean(poly_overlay_vals):.2f} {overlay_unit}")
            pm3.metric("Polygon min", f"{np.nanmin(poly_overlay_vals):.2f} {overlay_unit}")
            pm4.metric("Polygon max", f"{np.nanmax(poly_overlay_vals):.2f} {overlay_unit}")

            pm5, pm6 = st.columns(2)
            pm5.metric("Polygon mean coherence", f"{np.nanmean(poly_coh_vals):.3f}" if poly_coh_vals.size else "NA")
            pm6.metric("Polygon mean displacement", f"{np.nanmean(cumulative_disp_mm[valid_poly_mask]):.2f} mm")

            poly_sel_dates = dates[idx_start:idx_end + 1]
            poly_abs_mm = compute_polygon_mean_timeseries(str(TS_H5), idx_start, idx_end, poly_rows, poly_cols)
            poly_cum_mm = poly_abs_mm - poly_abs_mm[0]

            poly_dt_days = np.diff(poly_sel_dates.values.astype("datetime64[D]")).astype(int).astype(float)
            if len(poly_cum_mm) >= 2:
                poly_rate_mm_yr = np.diff(poly_cum_mm) / poly_dt_days * 365.25
                poly_rate_dates = poly_sel_dates[1:]
            else:
                poly_rate_mm_yr = np.array([])
                poly_rate_dates = []

            g1, g2 = st.columns(2)

            with g1:
                fig_poly_disp = go.Figure()
                fig_poly_disp.add_trace(go.Scatter(
                    x=poly_sel_dates,
                    y=poly_cum_mm,
                    mode="lines+markers",
                    name="Polygon mean cumulative displacement"
                ))
                fig_poly_disp.update_layout(
                    title="Polygon mean cumulative displacement vs date",
                    xaxis_title="Date",
                    yaxis_title="Cumulative displacement (mm)",
                    height=340,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_poly_disp, width="stretch")

            with g2:
                fig_poly_rate = go.Figure()
                fig_poly_rate.add_trace(go.Scatter(
                    x=poly_rate_dates,
                    y=poly_rate_mm_yr,
                    mode="lines+markers",
                    name="Polygon mean rate"
                ))
                fig_poly_rate.update_layout(
                    title="Polygon mean deformation rate vs date",
                    xaxis_title="Date",
                    yaxis_title="Rate (mm/year)",
                    height=340,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_poly_rate, width="stretch")

            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(
                    build_histogram(poly_overlay_vals, f"Polygon {overlay_name} distribution", overlay_unit),
                    width="stretch"
                )
            with h2:
                if poly_coh_vals.size:
                    st.plotly_chart(
                        build_histogram(poly_coh_vals, "Polygon coherence distribution", "coherence"),
                        width="stretch"
                    )
                else:
                    st.info("Polygon için coherence histogramı üretilemedi.")
else:
    st.info("Polygon analizi için haritadaki çizim araçlarından polygon veya rectangle çiz.")
