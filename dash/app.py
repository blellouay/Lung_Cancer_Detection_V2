import os
import sys
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = 'none'
matplotlib.rcParams['axes.facecolor'] = 'none'
from PIL import Image
import torch
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DEFAULT_CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD  = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE     = 224

CUSTOM_PRIORITY_COLUMNS = [
    "f1_macro",
    "recall_macro",
    "min_per_class_recall",
    "accuracy"
]

PREDICTION_MODEL_NAMES = {
    "CNNBaseline", "ResNetScratch", "ResNet18Scratch",
    "ResNet18", "ResNet18Transfer", "ResNetTransfer",
    "VGG16Scratch", "VGG16", "MobileNetV2Scratch",
    "EfficientNetB0Scratch", "EfficientNetB0", "EfficientNet",
    "InceptionV3Transfer", "InceptionV3", "Inception"
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LungScan AI — Model Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DESIGN SYSTEM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --bg:           #f0f4f8;
    --surface:      #ffffff;
    --surface-2:    #f7f9fb;
    --surface-3:    #eef2f7;
    --sidebar-bg:   #2563eb;
    --sidebar-sec:  #1d4ed8;

    --text-primary:   #0f172a;
    --text-secondary: #475569;
    --text-muted:     #94a3b8;
    --text-on-dark:   #ffffff;
    --text-on-dark-2: #dbeafe;

    --blue:       #2563eb;
    --blue-light: #eff6ff;
    --blue-glow:  rgba(37,99,235,0.15);
    --teal:       #0d9488;
    --teal-light: #f0fdfa;
    --amber:      #d97706;
    --amber-light:#fffbeb;
    --green:      #16a34a;
    --green-light:#f0fdf4;
    --red:        #dc2626;
    --red-light:  #fef2f2;
    --violet:     #7c3aed;

    --border:     #e2e8f0;
    --border-2:   #cbd5e1;

    --radius:     10px;
    --radius-sm:  6px;
    --radius-lg:  14px;
    --shadow-sm:  0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    --shadow-md:  0 4px 12px rgba(15,23,42,0.08), 0 2px 4px rgba(15,23,42,0.04);
    --shadow-lg:  0 12px 40px rgba(15,23,42,0.10), 0 4px 8px rgba(15,23,42,0.05);

    --font:  'DM Sans', sans-serif;
    --mono:  'DM Mono', monospace;
}

/* ── Base reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Main content container ── */
.block-container {
    max-width: 1480px !important;
    margin: 1.5rem auto !important;
    padding: 2rem 2.5rem 4rem !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-lg) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--sidebar-sec) 100%) !important;
    border-right: 1px solid #1e40af !important;
    box-shadow: 4px 0 18px rgba(15,23,42,0.12) !important;
}

section[data-testid="stSidebar"]::before {
    content: "";
    display: block;
    height: 3px;
    background: rgba(255,255,255,0.55);
    position: absolute;
    top: 0; left: 0; right: 0;
}

section[data-testid="stSidebar"] * {
    font-family: var(--font) !important;
    color: var(--text-on-dark) !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--text-on-dark) !important;
    padding: 6px 4px !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.12) !important;
    border-radius: 8px;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMetric label {
    color: var(--text-on-dark-2) !important;
    font-size: 12px !important;
}

/* Sidebar controls: keep dropdown text readable on the blue sidebar */
section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #bfdbfe !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: var(--text-primary) !important;
}

div[data-baseweb="popover"],
div[role="listbox"] {
    background: #ffffff !important;
    color: var(--text-primary) !important;
}

div[data-baseweb="popover"] *,
div[role="option"] {
    color: var(--text-primary) !important;
}

/* Keep Streamlit's sidebar collapse behavior, but make the control look like a real button. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    border-radius: 9px !important;
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    color: var(--blue) !important;
    box-shadow: 0 3px 10px rgba(15,23,42,0.14) !important;
    opacity: 1 !important;
    font-size: 0 !important;
    position: relative !important;
}

[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    background: var(--blue-light) !important;
    border-color: #93c5fd !important;
}

[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] * {
    color: var(--blue) !important;
    fill: var(--blue) !important;
    stroke: var(--blue) !important;
    font-size: 0 !important;
}

[data-testid="collapsedControl"]::before,
[data-testid="stSidebarCollapseButton"]::before {
    content: "" !important;
    position: absolute !important;
    width: 14px !important;
    height: 2px !important;
    border-radius: 2px !important;
    background: var(--blue) !important;
    transform: rotate(45deg) !important;
}

[data-testid="collapsedControl"]::after,
[data-testid="stSidebarCollapseButton"]::after {
    content: "" !important;
    position: absolute !important;
    width: 14px !important;
    height: 2px !important;
    border-radius: 2px !important;
    background: var(--blue) !important;
    transform: rotate(-45deg) !important;
}

section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--text-on-dark) !important;
    font-weight: 700 !important;
    font-size: 22px !important;
}

/* Sidebar brand title */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 16px;
    border-bottom: 1px solid rgba(255,255,255,0.22);
    margin-bottom: 16px;
}
.sidebar-brand-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.sidebar-brand-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-on-dark) !important;
    letter-spacing: -0.02em;
}
.sidebar-brand-sub {
    font-size: 11px;
    color: var(--text-on-dark-2) !important;
    margin-top: 1px;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 22px;
    margin-bottom: 28px;
    border-bottom: 1px solid var(--border);
}
.page-header-icon {
    width: 46px; height: 46px;
    border-radius: var(--radius);
    background: linear-gradient(135deg, var(--blue), var(--teal));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
    box-shadow: 0 4px 16px var(--blue-glow);
}
.page-header-text { flex: 1; }
.page-header-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.page-header-sub {
    font-size: 13.5px;
    color: var(--text-secondary);
    margin-top: 3px;
    font-weight: 400;
}
.page-header-stamp {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 13px;
    white-space: nowrap;
    flex-shrink: 0;
}

/* ── Metric cards ── */
.mc-grid { display: flex; gap: 14px; margin-bottom: 8px; }
.mc {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px 16px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    min-height: 96px;
}
.mc:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.mc-accent-bar {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: var(--radius) 0 0 var(--radius);
}
.mc-bg-orb {
    position: absolute;
    right: -6px; bottom: -10px;
    width: 64px; height: 64px;
    border-radius: 50%;
    opacity: 0.06;
    filter: blur(8px);
}
.mc-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.mc-value {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    line-height: 1;
}
.mc-sub {
    font-size: 11.5px;
    color: var(--text-muted);
    margin-top: 6px;
    font-weight: 400;
}

/* ── Alert / Info boxes ── */
.alert {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 13px 16px;
    border-radius: var(--radius-sm);
    font-size: 13.5px;
    font-weight: 500;
    line-height: 1.6;
    margin: 8px 0;
}
.alert-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.alert-text { color: inherit; }
.alert-text strong { font-weight: 700; }

.alert-warn {
    background: var(--amber-light);
    border: 1px solid rgba(217,119,6,0.25);
    border-left: 3px solid var(--amber);
    color: #92400e;
}
.alert-ok {
    background: var(--green-light);
    border: 1px solid rgba(22,163,74,0.25);
    border-left: 3px solid var(--green);
    color: #14532d;
}
.alert-info {
    background: var(--blue-light);
    border: 1px solid rgba(37,99,235,0.2);
    border-left: 3px solid var(--blue);
    color: #1e3a5f;
}
.alert-danger {
    background: var(--red-light);
    border: 1px solid rgba(220,38,38,0.2);
    border-left: 3px solid var(--red);
    color: #7f1d1d;
}

/* ── Section labels ── */
.sec-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 28px 0 14px;
}
.sec-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Status pills ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11.5px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.02em;
}
.pill-ok {
    background: var(--green-light);
    color: var(--green);
    border: 1px solid rgba(22,163,74,0.3);
}
.pill-bad {
    background: var(--red-light);
    color: var(--red);
    border: 1px solid rgba(220,38,38,0.25);
}
.pill-warn {
    background: var(--amber-light);
    color: var(--amber);
    border: 1px solid rgba(217,119,6,0.3);
}

/* ── Readiness row ── */
.readiness-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
    color: var(--text-secondary);
    font-weight: 500;
}
.readiness-row:last-child { border-bottom: none; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 2px solid var(--border) !important;
    background: transparent !important;
}
button[data-baseweb="tab"] {
    font-family: var(--font) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    padding: 9px 18px !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease;
    letter-spacing: -0.01em;
}
button[data-baseweb="tab"]:hover {
    color: var(--blue) !important;
    background: var(--blue-light) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--blue) !important;
    background: var(--blue-light) !important;
    font-weight: 700 !important;
    box-shadow: inset 0 -2px 0 var(--blue) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
    box-shadow: none !important;
    background: #ffffff !important;
}

/* ── Buttons ── */
button[kind="primary"] {
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    background: var(--blue) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #ffffff !important;
    padding: 10px 24px !important;
    box-shadow: 0 2px 8px var(--blue-glow) !important;
    transition: all 0.2s ease !important;
    letter-spacing: -0.01em !important;
}
button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}
button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px var(--blue-glow) !important;
    transform: translateY(-1px) !important;
}
button[kind="primary"]:hover *,
button[kind="primary"]:hover p,
button[kind="primary"]:hover span {
    color: #ffffff !important;
}
button[kind="secondary"] {
    font-family: var(--font) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-2) !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
}
button[kind="secondary"]:hover {
    background: var(--blue-light) !important;
    border-color: var(--blue) !important;
    color: var(--blue) !important;
}

/* ── Inputs / Selects ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border-2) !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: -0.01em !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border-radius: var(--radius) !important;
    border: 2px dashed var(--border-2) !important;
    background: var(--surface-2) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue) !important;
    background: var(--blue-light) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface-2) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font) !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
}

/* ── Markdown text readability ── */
[data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}
[data-testid="stMarkdownContainer"] strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] code {
    font-family: var(--mono) !important;
    font-size: 12.5px !important;
    background: var(--surface-3) !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    color: var(--blue) !important;
    border: 1px solid var(--border) !important;
}

/* ── st.write text ── */
[data-testid="stText"],
.stMarkdown p,
p {
    color: var(--text-secondary) !important;
    font-size: 14px !important;
}

/* ── Code blocks ── */
pre, code {
    font-family: var(--mono) !important;
    font-size: 12.5px !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── Bar charts ── */
.stBarChart, .stVegaLiteChart {
    background: var(--surface-2) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    padding: 8px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 24px 0 !important;
}

/* ── Caption ── */
[data-testid="stCaption"] small,
.stCaption small {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--text-muted) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }

/* ── Warnings / errors ── */
.stAlert {
    border-radius: var(--radius) !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
}

/* Final sidebar readability pass: keep all text on the blue area white. */
section[data-testid="stSidebar"] :where(
    p, span, label, div, small, strong,
    h1, h2, h3, h4, h5, h6,
    li, a
) {
    color: var(--text-on-dark) !important;
}

section[data-testid="stSidebar"] :where(svg, path) {
    color: var(--text-on-dark) !important;
    fill: currentColor !important;
    stroke: currentColor !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMetric label,
section[data-testid="stSidebar"] .sidebar-brand-sub {
    color: var(--text-on-dark-2) !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] input {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
    stroke: var(--text-primary) !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] * {
    color: var(--blue) !important;
    fill: var(--blue) !important;
    stroke: var(--blue) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────
def find_json_files(base_dir):
    json_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file in {"test_metrics.json", "metrics.json"}:
                json_files.append(os.path.join(root, file))
    return json_files


def load_single_result(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_project_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def get_class_names(data):
    d = data.get("deployment_info", {})
    c = data.get("training_config", {})
    return d.get("class_names") or c.get("class_names") or DEFAULT_CLASS_NAMES


def get_per_class_recall(metrics, class_names):
    pcr = metrics.get("per_class_recall")
    if pcr:
        return {cn: float(pcr.get(cn, 0)) for cn in class_names}
    cm = metrics.get("confusion_matrix")
    if not cm:
        return {}
    recalls = {}
    for i, cn in enumerate(class_names):
        if i >= len(cm):
            recalls[cn] = 0.0
            continue
        row_total = sum(cm[i])
        tp = cm[i][i] if i < len(cm[i]) else 0
        recalls[cn] = float(tp / row_total) if row_total else 0.0
    return recalls


def get_minimum_recall_info(metrics, class_names):
    pcr = get_per_class_recall(metrics, class_names)
    if not pcr:
        return None, None
    wc, wr = min(pcr.items(), key=lambda x: x[1])
    return float(wr), wc


def model_path_exists(data, result_path):
    d = data.get("deployment_info", {})
    mp = d.get("model_path")
    if mp is None:
        mp = os.path.join(os.path.dirname(result_path), "model.pth")
    else:
        mp = resolve_project_path(mp)
    return bool(mp and os.path.exists(mp))


def gradcam_available(data):
    d = data.get("deployment_info", {})
    m = data.get("evaluation_metrics", {})
    return bool(d.get("gradcam_available") or m.get("gradcam_paths"))


def prediction_demo_ready(data, result_path):
    mi = data.get("model_info", {})
    cn = get_class_names(data)
    return bool(model_path_exists(data, result_path) and cn and mi.get("model_name") in PREDICTION_MODEL_NAMES)


def load_results(json_files):
    rows = []
    for path in json_files:
        data = load_single_result(path)
        mi = data.get("model_info", {})
        me = data.get("evaluation_metrics", {})
        cf = data.get("training_config", {})
        if not mi or not me:
            continue
        cn = get_class_names(data)
        mr, wc = get_minimum_recall_info(me, cn)
        rd = os.path.dirname(path)
        hp  = os.path.join(rd, "training_history.json")
        lcp = os.path.join(rd, "loss_curve.png")
        acp = os.path.join(rd, "accuracy_curve.png")
        flp = os.path.join(rd, "finetune_loss_curve.png")
        fap = os.path.join(rd, "finetune_accuracy_curve.png")
        rows.append({
            "run_path": path, "run_dir": rd,
            "model_name": mi.get("model_name", "Unknown"),
            "accuracy": me.get("accuracy"),
            "precision_macro": me.get("precision_macro"),
            "recall_macro": me.get("recall_macro"),
            "f1_macro": me.get("f1_macro"),
            "min_per_class_recall": mr, "weakest_class": wc,
            "total_parameters": mi.get("total_parameters"),
            "trainable_parameters": mi.get("trainable_parameters"),
            "batch_size": cf.get("batch_size"), "epochs": cf.get("epochs"),
            "best_epoch": cf.get("best_epoch"), "learning_rate": cf.get("learning_rate"),
            "optimizer": cf.get("optimizer"), "loss_function": cf.get("loss_function"),
            "focal_gamma": cf.get("focal_gamma"), "sampler": cf.get("sampler"),
            "early_stopping": cf.get("early_stopping"),
            "model_pth_exists": model_path_exists(data, path),
            "class_names_saved": bool(cn),
            "gradcam_available": gradcam_available(data),
            "prediction_demo_ready": prediction_demo_ready(data, path),
            "history_path": hp if os.path.exists(hp) else None,
            "loss_curve_path": lcp if os.path.exists(lcp) else None,
            "accuracy_curve_path": acp if os.path.exists(acp) else None,
            "finetune_loss_curve_path": flp if os.path.exists(flp) else None,
            "finetune_accuracy_curve_path": fap if os.path.exists(fap) else None,
            "training_curves_available": os.path.exists(lcp) and os.path.exists(acp),
            "finetune_curves_available": os.path.exists(flp) and os.path.exists(fap),
        })
    return pd.DataFrame(rows)


def get_deployment_info(data, result_path):
    di = data.get("deployment_info") or {}
    mi = data.get("model_info", {})
    cf = data.get("training_config", {})
    input_size = di.get("input_size") or mi.get("input_size")
    image_size = di.get("image_size") or cf.get("image_size")
    if image_size is None and input_size:
        image_size = input_size[-1]
    mp = di.get("model_path")
    if mp is None:
        mp = os.path.join(os.path.dirname(result_path), "model.pth")
    else:
        mp = resolve_project_path(mp)
    return {
        **di,
        "model_path": mp,
        "input_size": input_size or [3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
        "image_size": image_size or DEFAULT_IMAGE_SIZE,
        "class_names": di.get("class_names") or cf.get("class_names") or DEFAULT_CLASS_NAMES,
        "normalize_mean": di.get("normalize_mean") or cf.get("normalize_mean") or DEFAULT_NORMALIZE_MEAN,
        "normalize_std":  di.get("normalize_std")  or cf.get("normalize_std")  or DEFAULT_NORMALIZE_STD,
    }


# ─────────────────────────────────────────────
# HELPER: Build a unique display label for each run
# e.g. "ResNet18 (run #2)" if there are duplicates
# ─────────────────────────────────────────────
def build_model_labels(df):
    """
    Returns a dict: run_path -> display_label
    Uses model_name; appends a counter when the same name appears multiple times.
    """
    name_counts = {}
    for name in df["model_name"]:
        name_counts[name] = name_counts.get(name, 0) + 1

    seen = {}
    labels = {}
    for _, row in df.iterrows():
        name = row["model_name"]
        path = row["run_path"]
        if name_counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            labels[path] = f"{name}  (run #{seen[name]})"
        else:
            labels[path] = name
    return labels


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def page_header(icon, title, subtitle):
    ts = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-icon">{icon}</div>
      <div class="page-header-text">
        <div class="page-header-title">{title}</div>
        <div class="page-header-sub">{subtitle}</div>
      </div>
      <span class="page-header-stamp">🕐 {ts}</span>
    </div>
    """, unsafe_allow_html=True)


# Accent colour palette
_ACCENTS = [
    ("#2563eb", "#eff6ff"),   # blue
    ("#0d9488", "#f0fdfa"),   # teal
    ("#d97706", "#fffbeb"),   # amber
    ("#16a34a", "#f0fdf4"),   # green
    ("#dc2626", "#fef2f2"),   # red
    ("#7c3aed", "#f5f3ff"),   # violet
]

def metric_card(label, value, color_idx=0, sub=None):
    color, bg = _ACCENTS[color_idx % len(_ACCENTS)]
    sub_html = f'<div class="mc-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="mc">
      <div class="mc-accent-bar" style="background:{color}"></div>
      <div class="mc-bg-orb" style="background:{color}"></div>
      <div class="mc-label">{label}</div>
      <div class="mc-value" style="color:{color}">{value}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def sec_label(text):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)

def dashboard_bar_chart(
    data,
    category_col,
    value_col,
    title=None,
    color="#2563eb",
    horizontal=False,
    height=320,
    value_format=",.2f",
    domain=None,
):
    chart_df = data[[category_col, value_col]].copy()
    chart_df = chart_df.dropna(subset=[category_col, value_col])
    if chart_df.empty:
        alert_info("No chart data available.")
        return

    chart_props = {"height": height}
    if title:
        chart_props["title"] = alt.TitleParams(
            text=title,
            anchor="start",
            color="#0f172a",
            font="DM Sans",
            fontSize=14,
            fontWeight=700,
            offset=12,
        )

    base = alt.Chart(chart_df).properties(**chart_props)
    axis_style = alt.Axis(
        labelColor="#475569",
        titleColor="#475569",
        labelFont="DM Sans",
        titleFont="DM Sans",
        gridColor="#e2e8f0",
        domainColor="#cbd5e1",
        tickColor="#cbd5e1",
        labelLimit=220,
    )
    scale = alt.Scale(domain=domain) if domain else alt.Undefined
    tooltip = [
        alt.Tooltip(f"{category_col}:N", title=category_col),
        alt.Tooltip(f"{value_col}:Q", title=value_col, format=value_format),
    ]

    if horizontal:
        bars = base.mark_bar(
            color=color,
            cornerRadiusEnd=6,
            size=18,
        ).encode(
            y=alt.Y(f"{category_col}:N", sort="-x", title=None, axis=axis_style),
            x=alt.X(f"{value_col}:Q", title=None, scale=scale, axis=axis_style),
            tooltip=tooltip,
        )
        labels = base.mark_text(
            align="left",
            baseline="middle",
            dx=6,
            color="#0f172a",
            font="DM Sans",
            fontSize=11,
            fontWeight=600,
        ).encode(
            y=alt.Y(f"{category_col}:N", sort="-x", title=None),
            x=alt.X(f"{value_col}:Q", scale=scale),
            text=alt.Text(f"{value_col}:Q", format=value_format),
        )
    else:
        bars = base.mark_bar(
            color=color,
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        ).encode(
            x=alt.X(f"{category_col}:N", sort=None, title=None, axis=axis_style),
            y=alt.Y(f"{value_col}:Q", title=None, scale=scale, axis=axis_style),
            tooltip=tooltip,
        )
        labels = base.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            color="#0f172a",
            font="DM Sans",
            fontSize=11,
            fontWeight=600,
        ).encode(
            x=alt.X(f"{category_col}:N", sort=None, title=None),
            y=alt.Y(f"{value_col}:Q", scale=scale),
            text=alt.Text(f"{value_col}:Q", format=value_format),
        )

    chart = (bars + labels).configure_view(
        stroke="#e2e8f0",
    ).configure_axis(
        labelFont="DM Sans",
        titleFont="DM Sans",
        grid=True,
    ).configure_title(
        font="DM Sans",
    )

    st.altair_chart(chart, use_container_width=True)


def alert_warn(msg):
    st.markdown(f'<div class="alert alert-warn"><span class="alert-icon">⚠️</span><span class="alert-text">{msg}</span></div>', unsafe_allow_html=True)

def alert_ok(msg):
    st.markdown(f'<div class="alert alert-ok"><span class="alert-icon">✅</span><span class="alert-text">{msg}</span></div>', unsafe_allow_html=True)

def alert_info(msg):
    st.markdown(f'<div class="alert alert-info"><span class="alert-icon">ℹ️</span><span class="alert-text">{msg}</span></div>', unsafe_allow_html=True)

def alert_danger(msg):
    st.markdown(f'<div class="alert alert-danger"><span class="alert-icon">🚨</span><span class="alert-text">{msg}</span></div>', unsafe_allow_html=True)


def medical_warning(recall):
    st.markdown("")
    if recall is None:
        alert_warn("Recall value not available — cannot assess clinical threshold.")
    elif recall < 0.70:
        alert_warn(f"<strong>Low recall ({recall:.4f})</strong> — this model may miss positive tumour cases in clinical use. Threshold: 0.70.")
    else:
        alert_ok(f"<strong>Recall {recall:.4f}</strong> meets the 0.70 clinical threshold. Model performance is acceptable.")
    st.markdown("")


def show_training_strategy(config):
    sec_label("Training Strategy")
    rows = [
        ["Loss function",  config.get("loss_function",  "—")],
        ["Focal gamma",    config.get("focal_gamma",    "—")],
        ["Sampler",        config.get("sampler",        "—")],
        ["Early stopping", config.get("early_stopping", "—")],
        ["Best epoch",     config.get("best_epoch",     "—")],
        ["Epochs",         config.get("epochs",         "—")],
        ["Optimizer",      config.get("optimizer",      "—")],
        ["Learning rate",  config.get("learning_rate",  "—")],
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Parameter", "Value"]),
        use_container_width=True, hide_index=True
    )


def show_deployment_readiness(data, result_path):
    sec_label("Deployment Readiness")
    cn = get_class_names(data)
    checks = [
        ("model.pth on disk",  model_path_exists(data, result_path)),
        ("class_names saved",  bool(cn)),
        ("Grad-CAM available", gradcam_available(data)),
        ("Prediction demo",    prediction_demo_ready(data, result_path)),
    ]
    for label, ok in checks:
        pill_html = (
            f'<span class="pill pill-ok">✓ Ready</span>' if ok
            else f'<span class="pill pill-bad">✗ Missing</span>'
        )
        st.markdown(
            f'<div class="readiness-row">'
            f'<span>{label}</span>{pill_html}'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown("")


def show_per_class_recall(metrics, class_names=None, threshold=0.50):
    sec_label("Per-Class Recall")
    class_names = class_names or DEFAULT_CLASS_NAMES
    pcr = get_per_class_recall(metrics, class_names)
    if not pcr:
        alert_info("No per-class recall found. Re-save evaluation results with per_class_recall or confusion_matrix.")
        return
    recall_df = pd.DataFrame(list(pcr.items()), columns=["Class", "Recall"])
    recall_df = recall_df.sort_values("Recall", ascending=True)
    recall_df["Recall (%)"] = (recall_df["Recall"] * 100).round(2)
    recall_df["Status"] = recall_df["Recall"].apply(lambda x: "⚠️ Low" if x < threshold else "✅ Good")
    weakest = recall_df.iloc[0]
    alert_warn(f"Weakest class: <strong>{weakest['Class']}</strong> — recall {weakest['Recall']:.4f}")
    st.markdown("")
    st.dataframe(recall_df[["Class", "Recall (%)", "Status"]], use_container_width=True, hide_index=True)
    dashboard_bar_chart(
        recall_df,
        "Class",
        "Recall (%)",
        title="Recall by Class",
        color="#2563eb",
        height=300,
        domain=[0, 100],
    )


def show_training_curves(selected_data):
    sec_label("Training Curves")

    def vp(p):
        return isinstance(p, (str, bytes, os.PathLike)) and os.path.exists(p)

    lc  = selected_data.get("loss_curve_path")
    ac  = selected_data.get("accuracy_curve_path")
    flc = selected_data.get("finetune_loss_curve_path")
    fac = selected_data.get("finetune_accuracy_curve_path")
    hp  = selected_data.get("history_path")

    if vp(hp):
        st.caption(f"History file: {hp}")

    c1, c2 = st.columns(2)
    with c1:
        if vp(lc):   st.image(lc, caption="Loss Curve", use_container_width=True)
        else:         alert_info("Loss curve image not available.")
    with c2:
        if vp(ac):   st.image(ac, caption="Accuracy Curve", use_container_width=True)
        else:         alert_info("Accuracy curve image not available.")

    if vp(flc) or vp(fac):
        sec_label("Fine-Tuning Curves")
        c3, c4 = st.columns(2)
        with c3:
            if vp(flc): st.image(flc, caption="Fine-Tuning Loss", use_container_width=True)
            else:        alert_info("Fine-tuning loss curve not available.")
        with c4:
            if vp(fac): st.image(fac, caption="Fine-Tuning Accuracy", use_container_width=True)
            else:        alert_info("Fine-tuning accuracy curve not available.")


def show_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="600", color="#0f172a", pad=14)
    ax.set_xlabel("Predicted", fontsize=11, color="#475569")
    ax.set_ylabel("True", fontsize=11, color="#475569")
    ax.tick_params(colors="#475569", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    max_val = max(max(r) for r in cm)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > max_val * 0.5 else "#0f172a",
                    fontsize=12, fontweight="600")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)


def show_architecture_section(model_info):
    sec_label("Model Architecture")
    model_name       = model_info.get("model_name", "Unknown")
    total_params     = model_info.get("total_parameters",    0)
    trainable_params = model_info.get("trainable_parameters", 0)
    input_size       = model_info.get("input_size")
    layers           = model_info.get("layers", [])

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Total Parameters",     f"{total_params:,}", 0)
    with c2: metric_card("Trainable Parameters", f"{trainable_params:,}", 1)
    with c3: metric_card("Input Size", str(input_size) if input_size else "N/A", 2)
    st.markdown("")

    if model_name == "CNNBaseline":
        sec_label("CNN Baseline Architecture")
        st.code(
            "Input CT Image\n"
            "     ↓\n"
            "Conv Block 1: Conv2D → BatchNorm → ReLU → MaxPool\n"
            "     ↓\n"
            "Conv Block 2: Conv2D → BatchNorm → ReLU → MaxPool\n"
            "     ↓\n"
            "Conv Block 3: Conv2D → BatchNorm → ReLU → MaxPool\n"
            "     ↓\n"
            "Flatten  →  Dense(32)  →  Dropout(0.4)  →  Output (4 classes)",
            language="text"
        )
        st.dataframe(pd.DataFrame([
            ["Conv Block 1", "32",  "Low-level feature extraction"],
            ["Conv Block 2", "64",  "Intermediate pattern detection"],
            ["Conv Block 3", "128", "High-level tumour feature extraction"],
            ["Flatten",       "—",  "Feature maps → vector"],
            ["Dense Layer",  "32",  "Classification features"],
            ["Dropout",      "0.4", "Reduce overfitting"],
            ["Output Layer",  "4",  "Tumour class prediction"],
        ], columns=["Layer", "Output Channels/Units", "Purpose"]),
            use_container_width=True, hide_index=True)
    else:
        alert_info(f"Auto-extracted architecture for <strong>{model_name}</strong>.")
        st.markdown("")
        if layers:
            sec_label("Layer Type Summary")
            tc = {}
            for l in layers:
                t = l.get("type", "Unknown")
                tc[t] = tc.get(t, 0) + 1
            st.dataframe(
                pd.DataFrame([{"Layer Type": k, "Count": v} for k, v in tc.items()])
                  .sort_values("Count", ascending=False),
                use_container_width=True, hide_index=True)

            sec_label("Full Layer Breakdown")
            cf_col, ct_col = st.columns([3, 1])
            with cf_col:
                all_types = sorted(set(l.get("type", "") for l in layers))
                selected_types = st.multiselect("Filter by layer type", options=all_types, default=all_types)
            with ct_col:
                show_zero = st.checkbox("Show zero-param layers", value=False)

            filtered = [l for l in layers if l.get("type", "") in selected_types
                        and (show_zero or l.get("parameters", 0) > 0)]
            if filtered:
                ld = pd.DataFrame(filtered)[
                    ["name","type","input_shape","output_shape","parameters","trainable_parameters"]
                ].rename(columns={
                    "name": "Layer Name", "type": "Type",
                    "input_shape": "Input Shape", "output_shape": "Output Shape",
                    "parameters": "Parameters", "trainable_parameters": "Trainable"
                })
                st.dataframe(ld, use_container_width=True, hide_index=True)
                sec_label("Parameter Distribution")
                pd2 = ld[ld["Parameters"] > 0]
                if not pd2.empty:
                    dashboard_bar_chart(
                        pd2,
                        "Layer Name",
                        "Parameters",
                        title="Parameters by Layer",
                        color="#0d9488",
                        horizontal=True,
                        height=max(280, min(520, len(pd2) * 28)),
                        value_format=",.0f",
                    )
            else:
                alert_info("No layers match the current filters.")
        else:
            alert_warn("No structured layer data found. Re-save results using the updated save_evaluation_results().")

    with st.expander("Full Technical Architecture", expanded=False):
        arch_img  = model_info.get("architecture_image")
        arch_text = model_info.get("architecture")
        if arch_img and os.path.exists(arch_img):
            if arch_img.endswith(".svg"):
                with open(arch_img, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=900, scrolling=True)
            else:
                st.image(arch_img, caption=f"{model_name} Architecture", use_container_width=True)
        elif arch_text:
            st.code(arch_text, language="text")
        else:
            alert_info("No architecture information available.")


def unique_columns(cols):
    return list(dict.fromkeys(cols))


def sort_by_custom_priority(df):
    return df.sort_values(by=CUSTOM_PRIORITY_COLUMNS, ascending=[False]*4, na_position="last")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
json_files = find_json_files(RESULTS_DIR)
if len(json_files) == 0:
    st.warning("No JSON evaluation files found in the results folder.")
    st.stop()

df = load_results(json_files)
if df.empty:
    st.warning("No valid evaluation result files found.")
    st.stop()

# Build model label map (name → path lookup)
model_labels = build_model_labels(df)   # path -> display label
label_to_path = {v: k for k, v in model_labels.items()}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-brand-icon">🫁</div>
  <div>
    <div class="sidebar-brand-name">LungScan AI</div>
    <div class="sidebar-brand-sub">CT Scan · Cancer Classification</div>
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["📊 Compare All Models", "🏆 Best Model", "🔍 Individual Model", "🚀 Prediction Demo"]
)

st.sidebar.markdown("---")

col_a, col_b = st.sidebar.columns(2)
col_a.metric("Runs", len(df))
col_b.metric("Best F1", f"{df['f1_macro'].max():.3f}" if not df['f1_macro'].isna().all() else "—")

st.sidebar.markdown("")
priority_metric_sidebar = st.sidebar.selectbox(
    "Priority metric",
    ["custom_medical_priority", "f1_macro", "recall_macro",
     "min_per_class_recall", "accuracy", "precision_macro"]
)


# ─────────────────────────────────────────────
# PAGE 1: COMPARE ALL MODELS
# ─────────────────────────────────────────────
if page == "📊 Compare All Models":
    page_header("📊", "Compare All Models", "Full overview of every saved evaluation run.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Runs",      str(len(df)),                                0)
    with c2: metric_card("Best Recall",     f"{df['recall_macro'].max():.4f}",           1)
    with c3: metric_card("Best F1",         f"{df['f1_macro'].max():.4f}",               2)
    with c4: metric_card("Best Min Recall", f"{df['min_per_class_recall'].max():.4f}",   3)
    with c5: metric_card("Best Accuracy",   f"{df['accuracy'].max():.4f}",               4)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Model Table", "🏅 Metric Ranking", "⚙️ Parameters"])

    with tab1:
        display_cols = [
            "model_name", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "min_per_class_recall", "weakest_class", "total_parameters",
            "batch_size", "epochs", "best_epoch", "learning_rate", "optimizer",
            "loss_function", "focal_gamma", "sampler", "early_stopping",
            "model_pth_exists", "class_names_saved", "gradcam_available", "prediction_demo_ready"
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    with tab2:
        metric_choice = st.selectbox("Choose metric", [
            "custom_medical_priority", "f1_macro", "recall_macro",
            "min_per_class_recall", "accuracy", "precision_macro"
        ])
        if metric_choice == "custom_medical_priority":
            sorted_df    = sort_by_custom_priority(df)
            chart_metric = "f1_macro"
            alert_info("Custom priority ranks by: macro F1 → macro recall → min per-class recall → accuracy.")
        else:
            sorted_df    = df.sort_values(by=metric_choice, ascending=False)
            chart_metric = metric_choice
        st.markdown("")
        dashboard_bar_chart(
            sorted_df,
            "model_name",
            chart_metric,
            title="Model Metric Ranking",
            color="#2563eb",
            horizontal=True,
            height=max(280, min(460, len(sorted_df) * 42)),
        )
        ranking_cols = unique_columns(["model_name", chart_metric, "accuracy",
                                       "precision_macro", "recall_macro", "f1_macro",
                                       "min_per_class_recall", "weakest_class"])
        st.dataframe(sorted_df[ranking_cols], use_container_width=True, hide_index=True)

    with tab3:
        param_df = df.sort_values("total_parameters", ascending=False)
        dashboard_bar_chart(
            param_df,
            "model_name",
            "total_parameters",
            title="Total Parameters by Model",
            color="#0d9488",
            horizontal=True,
            height=max(280, min(460, len(param_df) * 42)),
            value_format=",.0f",
        )
        st.dataframe(
            param_df[["model_name", "total_parameters", "trainable_parameters"]],
            use_container_width=True, hide_index=True
        )


# ─────────────────────────────────────────────
# PAGE 2: BEST MODEL
# ─────────────────────────────────────────────
elif page == "🏆 Best Model":
    page_header("🏆", "Best Model", "Selected by medical priority — not accuracy alone.")

    if priority_metric_sidebar == "custom_medical_priority":
        ranked_df      = sort_by_custom_priority(df)
        priority_label = "macro F1 + recall + min per-class recall"
    else:
        ranked_df      = df.sort_values(by=priority_metric_sidebar, ascending=False)
        priority_label = priority_metric_sidebar

    best_row      = ranked_df.iloc[0]
    selected_data = load_single_result(best_row["run_path"])
    mi = selected_data.get("model_info", {})
    me = selected_data.get("evaluation_metrics", {})
    cf = selected_data.get("training_config", {})

    alert_ok(f"Best model by <strong>{priority_label}</strong>: <strong>{best_row['model_name']}</strong>")
    st.markdown("")

    # Show ranking table using model names
    ranked_display = ranked_df[["model_name", "f1_macro", "recall_macro",
                                  "min_per_class_recall", "weakest_class", "accuracy"]].copy()
    st.dataframe(ranked_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Accuracy",        f"{me.get('accuracy', 0):.4f}",        0)
    with c2: metric_card("Precision Macro", f"{me.get('precision_macro', 0):.4f}", 1)
    with c3: metric_card("Recall Macro",    f"{me.get('recall_macro', 0):.4f}",    2)
    with c4: metric_card("F1 Macro",        f"{me.get('f1_macro', 0):.4f}",        3)
    mr, _ = get_minimum_recall_info(me, get_class_names(selected_data))
    with c5: metric_card("Min Recall", f"{mr:.4f}" if mr is not None else "N/A", 4)

    medical_warning(me.get("recall_macro"))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Details", "🔲 Confusion Matrix",
        "📊 Classification Report", "🏗️ Architecture", "📈 Training Curves"
    ])

    with tab1:
        lc, rc = st.columns(2)
        with lc:
            sec_label("Model Information")
            st.write(f"**Model:** {mi.get('model_name')}")
            st.write(f"**Total parameters:** {mi.get('total_parameters')}")
            st.write(f"**Trainable parameters:** {mi.get('trainable_parameters')}")
        with rc:
            show_training_strategy(cf)
            show_deployment_readiness(selected_data, best_row["run_path"])

    with tab2:
        if "confusion_matrix" in me: show_confusion_matrix(me["confusion_matrix"])
        else: alert_info("No confusion matrix available.")

    with tab3:
        if "classification_report" in me: st.code(me["classification_report"], language="text")
        else: alert_info("No classification report available.")
        show_per_class_recall(me, get_class_names(selected_data))

    with tab4:
        show_architecture_section(mi)

    with tab5:
        show_training_curves(best_row)


# ─────────────────────────────────────────────
# PAGE 3: INDIVIDUAL MODEL
# ─────────────────────────────────────────────
elif page == "🔍 Individual Model":
    page_header("🔍", "Individual Model Inspection", "Select a model by name and inspect all evaluation details.")

    # Model name dropdown — shows clean names, not file paths
    all_labels = [model_labels[path] for path in df["run_path"]]
    selected_label = st.selectbox("Choose a model", all_labels)
    selected_path  = label_to_path[selected_label]
    selected_row   = df[df["run_path"] == selected_path].iloc[0]
    selected_data  = load_single_result(selected_path)
    mi = selected_data.get("model_info", {})
    me = selected_data.get("evaluation_metrics", {})
    cf = selected_data.get("training_config", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Accuracy",        f"{me.get('accuracy', 0):.4f}",        0)
    with c2: metric_card("Precision Macro", f"{me.get('precision_macro', 0):.4f}", 1)
    with c3: metric_card("Recall Macro",    f"{me.get('recall_macro', 0):.4f}",    2)
    with c4: metric_card("F1 Macro",        f"{me.get('f1_macro', 0):.4f}",        3)
    mr, _ = get_minimum_recall_info(me, get_class_names(selected_data))
    with c5: metric_card("Min Recall", f"{mr:.4f}" if mr is not None else "N/A", 4)

    medical_warning(me.get("recall_macro"))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📄 Model Info", "🔲 Confusion Matrix", "📊 Report",
        "⚙️ Training Config", "🚦 Deployment", "🏗️ Architecture", "📈 Curves"
    ])

    with tab1:
        sec_label("Model Information")
        st.write(f"**Model name:** {mi.get('model_name')}")
        st.write(f"**Total parameters:** {mi.get('total_parameters')}")
        st.write(f"**Trainable parameters:** {mi.get('trainable_parameters')}")

    with tab2:
        if "confusion_matrix" in me: show_confusion_matrix(me["confusion_matrix"])
        else: alert_info("No confusion matrix available.")

    with tab3:
        if "classification_report" in me: st.code(me["classification_report"], language="text")
        else: alert_info("No classification report available.")
        show_per_class_recall(me, get_class_names(selected_data))

    with tab4:
        show_training_strategy(cf)

    with tab5:
        show_deployment_readiness(selected_data, selected_path)

    with tab6:
        show_architecture_section(mi)

    with tab7:
        show_training_curves(selected_row)


# ─────────────────────────────────────────────
# PAGE 4: PREDICTION DEMO
# ─────────────────────────────────────────────
elif page == "🚀 Prediction Demo":
    page_header("🚀", "Prediction Demo", "Upload a CT scan image and run inference with a trained model.")

    # Model name dropdown — shows clean names, not file paths
    all_labels     = [model_labels[path] for path in df["run_path"]]
    selected_label = st.selectbox("Choose a trained model", all_labels)
    selected_path  = label_to_path[selected_label]
    selected_data  = load_single_result(selected_path)
    mi = selected_data.get("model_info", {})
    di = get_deployment_info(selected_data, selected_path)

    sec_label("Selected Model Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Model name:** {mi.get('model_name')}")
        st.write(f"**Input size:** {di.get('input_size')}")
        st.write(f"**Image size:** {di.get('image_size')} px")
    with c2:
        st.write(f"**Classes:** {', '.join(di.get('class_names', []))}")
        st.write(f"**Normalize mean:** {di.get('normalize_mean')}")
        st.write(f"**Normalize std:** {di.get('normalize_std')}")

    show_deployment_readiness(selected_data, selected_path)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CT scan image (PNG / JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded CT Scan", use_container_width=True)
        alert_warn("<strong>Research / demo use only</strong> — not a medical diagnosis.")
        st.markdown("---")

        if st.button("🔬 Run Prediction", type="primary"):
            mp          = di.get("model_path")
            model_name  = mi.get("model_name")
            class_names = di.get("class_names") or DEFAULT_CLASS_NAMES
            image_size  = di.get("image_size")  or DEFAULT_IMAGE_SIZE
            mean        = di.get("normalize_mean") or DEFAULT_NORMALIZE_MEAN
            std         = di.get("normalize_std")  or DEFAULT_NORMALIZE_STD

            if mp is None or not os.path.exists(mp):
                alert_danger("Model weights file not found. Make sure <code>model.pth</code> exists in the run directory.")
            else:
                import importlib
                import src.deployment.model_loader as model_loader_module
                from src.deployment.predict import get_prediction_transform, predict_image
                from src.interpretability.gradcam import generate_gradcam

                model_loader_module = importlib.reload(model_loader_module)
                load_trained_model  = model_loader_module.load_trained_model

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model  = load_trained_model(
                    model_name=model_name, model_path=mp,
                    num_classes=len(class_names), device=device
                )
                transform  = get_prediction_transform(image_size=image_size, mean=mean, std=std)
                prediction = predict_image(
                    model=model, image=image, transform=transform,
                    class_names=class_names, device=device
                )

                predicted_class  = prediction["predicted_class"]
                confidence       = prediction["confidence"]
                probabilities    = prediction["probabilities"]
                image_tensor     = prediction["image_tensor"]
                predicted_index  = prediction["predicted_index"]

                alert_ok("Prediction completed successfully.")
                st.markdown("")

                c1, c2 = st.columns(2)
                with c1: metric_card("Predicted Class", predicted_class,         0)
                with c2: metric_card("Confidence",      f"{confidence*100:.2f}%", 2)

                sec_label("Class Probabilities")
                prob_df = pd.DataFrame(list(probabilities.items()), columns=["Class", "Probability"])
                prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
                st.dataframe(prob_df[["Class", "Probability (%)"]], use_container_width=True, hide_index=True)
                dashboard_bar_chart(
                    prob_df,
                    "Class",
                    "Probability (%)",
                    title="Prediction Probability by Class",
                    color="#2563eb",
                    height=300,
                    domain=[0, 100],
                )

                sec_label("Grad-CAM Heatmap")
                gcdir = "dashboard_gradcam_outputs"
                os.makedirs(gcdir, exist_ok=True)
                gc_path = os.path.join(gcdir, "uploaded_image_gradcam.png")
                generate_gradcam(
                    model=model, image_tensor=image_tensor,
                    target_class=predicted_index, save_path=gc_path,
                    device=device, mean=mean, std=std
                )

                c3, c4 = st.columns(2)
                with c3: st.image(image,   caption="Original CT Scan",  use_container_width=True)
                with c4: st.image(gc_path, caption="Grad-CAM Heatmap",  use_container_width=True)
