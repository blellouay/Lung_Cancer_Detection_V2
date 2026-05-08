import os
import sys
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torch


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
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224

CUSTOM_PRIORITY_COLUMNS = [
    "f1_macro",
    "recall_macro",
    "min_per_class_recall",
    "accuracy"
]

PREDICTION_MODEL_NAMES = {
    "CNNBaseline",
    "ResNetScratch",
    "ResNet18Scratch",
    "ResNet18",
    "ResNet18Transfer",
    "ResNetTransfer",
    "VGG16Scratch",
    "VGG16",
    "MobileNetV2Scratch",
    "EfficientNetB0Scratch",
    "EfficientNetB0",
    "EfficientNet",
    "InceptionV3Transfer",
    "InceptionV3",
    "Inception"
}


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Tumor Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
    color: #ffffff;
}

.subtitle {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 25px;
}

.card {
    padding: 22px;
    border-radius: 18px;
    background-color: #ffffff;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    color: #111827;
}

.metric-label {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #111827;
}

.warning-box {
    padding: 18px;
    border-radius: 14px;
    background-color: #fff7ed;
    border: 1px solid #fdba74;
    color: #9a3412;
    font-weight: 600;
}

.success-box {
    padding: 18px;
    border-radius: 14px;
    background-color: #ecfdf5;
    border: 1px solid #6ee7b7;
    color: #065f46;
    font-weight: 600;
}

.info-box {
    padding: 18px;
    border-radius: 14px;
    background-color: #eff6ff;
    border: 1px solid #93c5fd;
    color: #1e40af;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# DATA FUNCTIONS
# =========================
def find_json_files(base_dir):
    json_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
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
    deployment_info = data.get("deployment_info", {})
    config = data.get("training_config", {})

    return (
        deployment_info.get("class_names")
        or config.get("class_names")
        or DEFAULT_CLASS_NAMES
    )


def get_per_class_recall(metrics, class_names):
    per_class_recall = metrics.get("per_class_recall")

    if per_class_recall:
        return {
            class_name: float(per_class_recall.get(class_name, 0))
            for class_name in class_names
        }

    cm = metrics.get("confusion_matrix")
    if not cm:
        return {}

    recalls = {}
    for i, class_name in enumerate(class_names):
        if i >= len(cm):
            recalls[class_name] = 0.0
            continue

        row_total = sum(cm[i])
        true_positive = cm[i][i] if i < len(cm[i]) else 0
        recalls[class_name] = float(true_positive / row_total) if row_total else 0.0

    return recalls


def get_minimum_recall_info(metrics, class_names):
    per_class_recall = get_per_class_recall(metrics, class_names)

    if not per_class_recall:
        return None, None

    weakest_class, weakest_recall = min(
        per_class_recall.items(),
        key=lambda item: item[1]
    )

    return float(weakest_recall), weakest_class


def model_path_exists(data, result_path):
    deployment_info = data.get("deployment_info", {})
    model_path = deployment_info.get("model_path")

    if model_path is None:
        model_path = os.path.join(os.path.dirname(result_path), "model.pth")
    else:
        model_path = resolve_project_path(model_path)

    return bool(model_path and os.path.exists(model_path))


def gradcam_available(data):
    deployment_info = data.get("deployment_info", {})
    metrics = data.get("evaluation_metrics", {})

    return bool(
        deployment_info.get("gradcam_available")
        or metrics.get("gradcam_paths")
    )


def prediction_demo_ready(data, result_path):
    model_info = data.get("model_info", {})
    class_names = get_class_names(data)
    model_name = model_info.get("model_name")

    return bool(
        model_path_exists(data, result_path)
        and class_names
        and model_name in PREDICTION_MODEL_NAMES
    )


def load_results(json_files):
    rows = []

    for path in json_files:
        data = load_single_result(path)

        model_info = data.get("model_info", {})
        metrics = data.get("evaluation_metrics", {})
        config = data.get("training_config", {})
        class_names = get_class_names(data)
        min_recall, weakest_class = get_minimum_recall_info(metrics, class_names)

        # Result folder containing test_metrics.json
        run_dir = os.path.dirname(path)

        history_path = os.path.join(run_dir, "training_history.json")
        loss_curve_path = os.path.join(run_dir, "loss_curve.png")
        accuracy_curve_path = os.path.join(run_dir, "accuracy_curve.png")

        rows.append({
            "run_path": path,
            "run_dir": run_dir,

            "model_name": model_info.get("model_name", "Unknown"),

            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),

            "min_per_class_recall": min_recall,
            "weakest_class": weakest_class,

            "total_parameters": model_info.get("total_parameters"),
            "trainable_parameters": model_info.get("trainable_parameters"),

            "batch_size": config.get("batch_size"),
            "epochs": config.get("epochs"),
            "best_epoch": config.get("best_epoch"),
            "learning_rate": config.get("learning_rate"),
            "optimizer": config.get("optimizer"),
            "loss_function": config.get("loss_function"),
            "focal_gamma": config.get("focal_gamma"),
            "sampler": config.get("sampler"),
            "early_stopping": config.get("early_stopping"),

            "model_pth_exists": model_path_exists(data, path),
            "class_names_saved": bool(class_names),
            "gradcam_available": gradcam_available(data),
            "prediction_demo_ready": prediction_demo_ready(data, path),

            # Training curve files
            "history_path": history_path if os.path.exists(history_path) else None,
            "loss_curve_path": loss_curve_path if os.path.exists(loss_curve_path) else None,
            "accuracy_curve_path": accuracy_curve_path if os.path.exists(accuracy_curve_path) else None,
            "training_curves_available": (
                os.path.exists(loss_curve_path)
                and os.path.exists(accuracy_curve_path)
            )
        })

    return pd.DataFrame(rows)


def get_deployment_info(data, result_path):
    deployment_info = data.get("deployment_info") or {}
    model_info = data.get("model_info", {})
    config = data.get("training_config", {})

    input_size = deployment_info.get("input_size") or model_info.get("input_size")
    image_size = deployment_info.get("image_size") or config.get("image_size")

    if image_size is None and input_size:
        image_size = input_size[-1]

    model_path = deployment_info.get("model_path")
    if model_path is None:
        model_path = os.path.join(os.path.dirname(result_path), "model.pth")
    else:
        model_path = resolve_project_path(model_path)

    return {
        **deployment_info,
        "model_path": model_path,
        "input_size": input_size or [3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
        "image_size": image_size or DEFAULT_IMAGE_SIZE,
        "class_names": (
            deployment_info.get("class_names")
            or config.get("class_names")
            or DEFAULT_CLASS_NAMES
        ),
        "normalize_mean": (
            deployment_info.get("normalize_mean")
            or config.get("normalize_mean")
            or DEFAULT_NORMALIZE_MEAN
        ),
        "normalize_std": (
            deployment_info.get("normalize_std")
            or config.get("normalize_std")
            or DEFAULT_NORMALIZE_STD
        ),
    }


# =========================
# UI FUNCTIONS
# =========================
def show_training_curves(selected_data):
    st.subheader("Training Curves")

    loss_curve = selected_data.get("loss_curve_path")
    acc_curve = selected_data.get("accuracy_curve_path")

    col1, col2 = st.columns(2)

    with col1:
        if loss_curve and os.path.exists(loss_curve):
            st.image(loss_curve, caption="Loss Curve", use_container_width=True)
        else:
            st.info("Loss curve not available.")

    with col2:
        if acc_curve and os.path.exists(acc_curve):
            st.image(acc_curve, caption="Accuracy Curve", use_container_width=True)
        else:
            st.info("Accuracy curve not available.")
            
def show_per_class_recall(metrics, class_names=None, threshold=0.50):
    st.subheader("Per-Class Recall")

    class_names = class_names or DEFAULT_CLASS_NAMES
    per_class_recall = get_per_class_recall(metrics, class_names)

    if not per_class_recall:
        st.info("No per-class recall found. Re-save evaluation results with per_class_recall or confusion_matrix.")
        return

    recall_df = pd.DataFrame(
        list(per_class_recall.items()),
        columns=["Class", "Recall"]
    )
    recall_df = recall_df.sort_values("Recall", ascending=True)

    recall_df["Recall (%)"] = recall_df["Recall"] * 100
    recall_df["Status"] = recall_df["Recall"].apply(
        lambda x: "⚠️ Low" if x < threshold else "✅ Good"
    )

    weakest = recall_df.iloc[0]
    st.warning(
        f"Weakest class: {weakest['Class']} "
        f"({weakest['Recall']:.4f} recall)."
    )

    st.dataframe(
        recall_df[["Class", "Recall (%)", "Status"]],
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        recall_df.set_index("Class")["Recall (%)"]
    )

def title_block(title, subtitle):
    st.markdown(f"<div class='main-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def unique_columns(cols):
    return list(dict.fromkeys(cols))


def sort_by_custom_priority(dataframe):
    return dataframe.sort_values(
        by=CUSTOM_PRIORITY_COLUMNS,
        ascending=[False, False, False, False],
        na_position="last"
    )


def show_training_strategy(config):
    st.subheader("Training Strategy")

    strategy_df = pd.DataFrame(
        [
            ["Loss", config.get("loss_function", "Not saved")],
            ["Focal gamma", config.get("focal_gamma", "Not saved")],
            ["Sampler", config.get("sampler", "Not saved")],
            ["Early stopping", config.get("early_stopping", "Not saved")],
            ["Best epoch", config.get("best_epoch", "Not saved")],
            ["Epochs", config.get("epochs", "Not saved")],
            ["Optimizer", config.get("optimizer", "Not saved")],
            ["Learning rate", config.get("learning_rate", "Not saved")],
        ],
        columns=["Item", "Value"]
    )

    st.dataframe(strategy_df, use_container_width=True, hide_index=True)


def show_deployment_readiness(data, result_path):
    st.subheader("Deployment Readiness")

    class_names = get_class_names(data)
    readiness_df = pd.DataFrame(
        [
            ["model.pth exists", model_path_exists(data, result_path)],
            ["class_names saved", bool(class_names)],
            ["Grad-CAM available", gradcam_available(data)],
            ["prediction demo ready", prediction_demo_ready(data, result_path)],
        ],
        columns=["Check", "Ready"]
    )

    readiness_df["Status"] = readiness_df["Ready"].apply(
        lambda ready: "Ready" if ready else "Missing"
    )

    st.dataframe(
        readiness_df[["Check", "Status"]],
        use_container_width=True,
        hide_index=True
    )


def medical_warning(recall):
    if recall is None:
        st.warning("Recall value not available.")
        return

    if recall < 0.70:
        st.markdown(
            """
            <div class="warning-box">
                ⚠️ Low recall detected. In medical tumor detection, this model may miss positive tumor cases.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success-box">
                ✅ Recall is acceptable compared to the chosen threshold.
            </div>
            """,
            unsafe_allow_html=True
        )


def show_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    st.pyplot(fig)


def show_architecture_section(model_info):
    st.subheader("Model Architecture")

    model_name = model_info.get("model_name", "Unknown Model")
    layers = model_info.get("layers", [])

    # --- Summary metrics row ---
    total_params = model_info.get("total_parameters", 0)
    trainable_params = model_info.get("trainable_parameters", 0)
    input_size = model_info.get("input_size")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Parameters", f"{total_params:,}")
    with col2:
        metric_card("Trainable Parameters", f"{trainable_params:,}")
    with col3:
        metric_card("Input Size", str(input_size) if input_size else "N/A")

    st.markdown("")

    # --- CNNBaseline: hardcoded diagram + table ---
    if model_name == "CNNBaseline":
        st.markdown("### CNN Baseline Overview")

        st.markdown("""
```text
Input CT Image
     ↓
Conv Block 1: Conv2D → BatchNorm → ReLU → MaxPool
     ↓
Conv Block 2: Conv2D → BatchNorm → ReLU → MaxPool
     ↓
Conv Block 3: Conv2D → BatchNorm → ReLU → MaxPool
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Dropout
     ↓
Output Layer: 4 classes
```
""")

        architecture_table = pd.DataFrame(
            [
                ["Conv Block 1", "32", "Low-level feature extraction"],
                ["Conv Block 2", "64", "Intermediate pattern detection"],
                ["Conv Block 3", "128", "High-level tumor feature extraction"],
                ["Flatten", "-", "Convert feature maps to vector"],
                ["Dense Layer", "32", "Classification features"],
                ["Dropout", "0.4", "Reduce overfitting"],
                ["Output Layer", "4", "Tumor class prediction"],
            ],
            columns=["Layer", "Output Channels / Units", "Purpose"]
        )

        st.dataframe(
            architecture_table,
            use_container_width=True,
            hide_index=True
        )

    # --- Any other model: dynamic layer table from JSON ---
    else:
        st.markdown(
            f"""
            <div class="info-box">
                Showing auto-extracted architecture for <b>{model_name}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        if layers:
            # --- Layer type summary ---
            st.markdown("#### Layer Type Summary")

            type_counts = {}
            for layer in layers:
                t = layer.get("type", "Unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            summary_df = pd.DataFrame(
                [{"Layer Type": k, "Count": v} for k, v in type_counts.items()]
            ).sort_values("Count", ascending=False)

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("#### Full Layer Breakdown")

            # --- Filters ---
            col_filter, col_toggle = st.columns([3, 1])

            with col_filter:
                all_types = sorted(set(l.get("type", "") for l in layers))
                selected_types = st.multiselect(
                    "Filter by layer type",
                    options=all_types,
                    default=all_types
                )

            with col_toggle:
                show_zero_params = st.checkbox("Show zero-param layers", value=False)

            # --- Build filtered dataframe ---
            filtered = [
                l for l in layers
                if l.get("type", "") in selected_types
                and (show_zero_params or l.get("parameters", 0) > 0)
            ]

            if filtered:
                layers_df = pd.DataFrame(filtered)[[
                    "name",
                    "type",
                    "input_shape",
                    "output_shape",
                    "parameters",
                    "trainable_parameters"
                ]].rename(columns={
                    "name": "Layer Name",
                    "type": "Type",
                    "input_shape": "Input Shape",
                    "output_shape": "Output Shape",
                    "parameters": "Parameters",
                    "trainable_parameters": "Trainable Params"
                })

                st.dataframe(layers_df, use_container_width=True, hide_index=True)

                # --- Parameter distribution bar chart ---
                st.markdown("#### Parameter Distribution")

                param_df = layers_df[layers_df["Parameters"] > 0].copy()

                if not param_df.empty:
                    st.bar_chart(
                        param_df.set_index("Layer Name")["Parameters"]
                    )

            else:
                st.info("No layers match the current filters.")

        else:
            st.markdown(
                """
                <div class="warning-box">
                    ⚠️ No structured layer data found. Re-save your results using the
                    updated save_evaluation_results() function to enable this view.
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Full technical expander (SVG image or raw text) ---
    with st.expander("Show Full Technical Architecture", expanded=False):
        architecture_image = model_info.get("architecture_image")
        architecture_text = model_info.get("architecture")

        if architecture_image and os.path.exists(architecture_image):
            if architecture_image.endswith(".svg"):
                with open(architecture_image, "r", encoding="utf-8") as f:
                    svg_content = f.read()

                st.components.v1.html(
                    svg_content,
                    height=900,
                    scrolling=True
                )
            else:
                st.image(
                    architecture_image,
                    caption=f"{model_name} Architecture",
                    use_container_width=True
                )

        elif architecture_text:
            st.code(architecture_text, language="text")

        else:
            st.info("No architecture information available.")


# =========================
# LOAD DATA
# =========================

json_files = find_json_files(RESULTS_DIR)

if len(json_files) == 0:
    st.warning("No JSON evaluation files found in the results folder.")
    st.stop()

df = load_results(json_files)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🧠 Tumor Dashboard")
st.sidebar.caption("PyTorch CT Scan Model Evaluation")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Compare All Models",
        "🏆 Best Model",
        "🔍 Individual Model",
        "🚀 Prediction Demo"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write(f"Detected runs: {len(df)}")

priority_metric_sidebar = st.sidebar.selectbox(
    "Medical priority metric",
    [
        "custom_medical_priority",
        "f1_macro",
        "recall_macro",
        "min_per_class_recall",
        "accuracy",
        "precision_macro"
    ]
)

# =========================
# PAGE 1: COMPARE ALL MODELS
# =========================

if page == "📊 Compare All Models":

    title_block(
        "Compare All Models",
        "Overview of all saved model evaluation results."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card("Total Runs", len(df))

    with col2:
        metric_card("Best Recall", f"{df['recall_macro'].max():.4f}")

    with col3:
        metric_card("Best F1", f"{df['f1_macro'].max():.4f}")

    with col4:
        metric_card("Best Min Recall", f"{df['min_per_class_recall'].max():.4f}")

    with col5:
        metric_card("Best Accuracy", f"{df['accuracy'].max():.4f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Model Table",
        "Metric Ranking",
        "Parameter Comparison"
    ])

    with tab1:
        display_cols = [
            "model_name",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "min_per_class_recall",
            "weakest_class",
            "total_parameters",
            "batch_size",
            "epochs",
            "best_epoch",
            "learning_rate",
            "optimizer",
            "loss_function",
            "focal_gamma",
            "sampler",
            "early_stopping",
            "model_pth_exists",
            "class_names_saved",
            "gradcam_available",
            "prediction_demo_ready"
        ]

        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        metric_choice = st.selectbox(
            "Choose metric",
            [
                "custom_medical_priority",
                "f1_macro",
                "recall_macro",
                "min_per_class_recall",
                "accuracy",
                "precision_macro"
            ]
        )

        if metric_choice == "custom_medical_priority":
            sorted_df = sort_by_custom_priority(df)
            chart_metric = "f1_macro"
            st.caption(
                "Custom priority ranks by macro F1, then macro recall, "
                "then minimum per-class recall, then accuracy."
            )
        else:
            sorted_df = df.sort_values(by=metric_choice, ascending=False)
            chart_metric = metric_choice

        st.bar_chart(
            sorted_df.set_index("model_name")[chart_metric]
        )

        ranking_cols = unique_columns([
            "model_name",
            chart_metric,
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "min_per_class_recall",
            "weakest_class"
        ])

        st.dataframe(
            sorted_df[ranking_cols],
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        param_df = df.sort_values(by="total_parameters", ascending=False)

        st.bar_chart(
            param_df.set_index("model_name")["total_parameters"]
        )

        st.dataframe(
            param_df[
                [
                    "model_name",
                    "total_parameters",
                    "trainable_parameters"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

# =========================
# PAGE 2: BEST MODEL
# =========================

elif page == "🏆 Best Model":

    title_block(
        "Best Model",
        "The best model is selected by medical priority, not accuracy alone."
    )

    if priority_metric_sidebar == "custom_medical_priority":
        ranked_df = sort_by_custom_priority(df)
        priority_label = "macro F1, macro recall, minimum per-class recall"
    else:
        ranked_df = df.sort_values(by=priority_metric_sidebar, ascending=False)
        priority_label = priority_metric_sidebar

    best_row = ranked_df.iloc[0]

    selected_data = load_single_result(best_row["run_path"])

    model_info = selected_data.get("model_info", {})
    metrics = selected_data.get("evaluation_metrics", {})
    config = selected_data.get("training_config", {})

    st.markdown(
        f"""
        <div class="success-box">
            Best model based on <b>{priority_label}</b>:
            {best_row['model_name']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    st.caption(
        "Custom priority ranks by macro F1, then macro recall, "
        "then minimum per-class recall, then accuracy."
    )

    st.dataframe(
        ranked_df[
            [
                "model_name",
                "f1_macro",
                "recall_macro",
                "min_per_class_recall",
                "weakest_class",
                "accuracy"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

    with col2:
        metric_card("Precision Macro", f"{metrics.get('precision_macro', 0):.4f}")

    with col3:
        metric_card("Recall Macro", f"{metrics.get('recall_macro', 0):.4f}")

    with col4:
        metric_card("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")

    min_recall, _ = get_minimum_recall_info(
        metrics,
        get_class_names(selected_data)
    )
    with col5:
        metric_card(
            "Minimum Recall",
            f"{min_recall:.4f}" if min_recall is not None else "N/A"
        )

    st.markdown("---")
    medical_warning(metrics.get("recall_macro"))
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Details",
        "Confusion Matrix",
        "Classification Report",
        "Architecture",
        "Training Curves"
    ])

    with tab1:
        left, right = st.columns(2)

        with left:
            st.subheader("Model Info")
            st.write(f"**Model name:** {model_info.get('model_name')}")
            st.write(f"**Total parameters:** {model_info.get('total_parameters')}")
            st.write(f"**Trainable parameters:** {model_info.get('trainable_parameters')}")
            st.write(f"**Result file:** {best_row['run_path']}")

        with right:
            show_training_strategy(config)
            show_deployment_readiness(selected_data, best_row["run_path"])

    with tab2:
        if "confusion_matrix" in metrics:
            show_confusion_matrix(metrics["confusion_matrix"])
        else:
            st.info("No confusion matrix available.")

    with tab3:
        if "classification_report" in metrics:
            st.text(metrics["classification_report"])
        else:
            st.info("No classification report available.")
        show_per_class_recall(metrics, get_class_names(selected_data))

    with tab4:
        show_architecture_section(model_info)

    with tab5:
        show_training_curves(best_row)


# =========================
# PAGE 3: INDIVIDUAL MODEL
# =========================

elif page == "🔍 Individual Model":

    title_block(
        "Individual Model Inspection",
        "Select one saved evaluation file and inspect all its details."
    )

    selected_path = st.selectbox(
        "Choose result file",
        df["run_path"].tolist()
    )

    selected_data = load_single_result(selected_path)

    model_info = selected_data.get("model_info", {})
    metrics = selected_data.get("evaluation_metrics", {})
    config = selected_data.get("training_config", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

    with col2:
        metric_card("Precision Macro", f"{metrics.get('precision_macro', 0):.4f}")

    with col3:
        metric_card("Recall Macro", f"{metrics.get('recall_macro', 0):.4f}")

    with col4:
        metric_card("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")

    min_recall, _ = get_minimum_recall_info(
        metrics,
        get_class_names(selected_data)
    )
    with col5:
        metric_card(
            "Minimum Recall",
            f"{min_recall:.4f}" if min_recall is not None else "N/A"
        )

    st.markdown("---")
    medical_warning(metrics.get("recall_macro"))
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 , tab7= st.tabs([
        "Model Info",
        "Confusion Matrix",
        "Classification Report",
        "Training Config",
        "Deployment",
        "Architecture",
        "Training Curves"

    ])

    with tab1:
        st.subheader("Model Information")
        st.write(f"**Model name:** {model_info.get('model_name')}")
        st.write(f"**Total parameters:** {model_info.get('total_parameters')}")
        st.write(f"**Trainable parameters:** {model_info.get('trainable_parameters')}")
        st.write(f"**Result path:** {selected_path}")

    with tab2:
        if "confusion_matrix" in metrics:
            show_confusion_matrix(metrics["confusion_matrix"])
        else:
            st.info("No confusion matrix available.")

    with tab3:
        if "classification_report" in metrics:
            st.text(metrics["classification_report"])
        else:
            st.info("No classification report available.")
        show_per_class_recall(metrics, get_class_names(selected_data))

    with tab4:
        show_training_strategy(config)

    with tab5:
        show_deployment_readiness(selected_data, selected_path)

    with tab6:
        show_architecture_section(model_info)
        
    with tab5:
        show_training_curves(best_row)

# =========================
# PAGE 4: PREDICTION DEMO
# =========================

elif page == "🚀 Prediction Demo":

    title_block(
        "Prediction Demo",
        "Upload a CT scan image and test a trained model."
    )

    selected_path = st.selectbox(
        "Choose trained model result file",
        df["run_path"].tolist()
    )

    selected_data = load_single_result(selected_path)

    model_info = selected_data.get("model_info", {})
    deployment_info = get_deployment_info(selected_data, selected_path)

    st.markdown("### Selected Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Model name:** {model_info.get('model_name')}")
        st.write(f"**Model path:** {deployment_info.get('model_path')}")
        st.write(f"**Input size:** {deployment_info.get('input_size')}")

    with col2:
        st.write(f"**Class names:** {deployment_info.get('class_names')}")
        st.write(f"**Mean:** {deployment_info.get('normalize_mean')}")
        st.write(f"**Std:** {deployment_info.get('normalize_std')}")

    show_deployment_readiness(selected_data, selected_path)

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CT scan image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded CT Scan",
            use_container_width=True
        )

        st.markdown(
            """
            <div class="warning-box">
                ⚠️ This prediction is for research/demo purposes only, not medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        if st.button("Run Prediction"):

            model_path = deployment_info.get("model_path")
            model_name = model_info.get("model_name")
            class_names = deployment_info.get("class_names") or DEFAULT_CLASS_NAMES
            image_size = deployment_info.get("image_size") or DEFAULT_IMAGE_SIZE
            mean = deployment_info.get("normalize_mean") or DEFAULT_NORMALIZE_MEAN
            std = deployment_info.get("normalize_std") or DEFAULT_NORMALIZE_STD

            if model_path is None or not os.path.exists(model_path):
                st.error("Model weights file not found. Make sure model.pth exists.")

            else:
                import importlib
                import src.deployment.model_loader as model_loader_module
                from src.deployment.predict import get_prediction_transform, predict_image
                from src.interpretability.gradcam import generate_gradcam

                model_loader_module = importlib.reload(model_loader_module)
                load_trained_model = model_loader_module.load_trained_model

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = load_trained_model(
                    model_name=model_name,
                    model_path=model_path,
                    num_classes=len(class_names),
                    device=device
                )

                transform = get_prediction_transform(
                    image_size=image_size,
                    mean=mean,
                    std=std
                )

                prediction = predict_image(
                    model=model,
                    image=image,
                    transform=transform,
                    class_names=class_names,
                    device=device
                )

                predicted_class = prediction["predicted_class"]
                confidence = prediction["confidence"]
                probabilities = prediction["probabilities"]
                image_tensor = prediction["image_tensor"]
                predicted_index = prediction["predicted_index"]

                st.success("Prediction completed successfully.")

                col1, col2 = st.columns(2)

                with col1:
                    metric_card("Predicted Class", predicted_class)

                with col2:
                    metric_card("Confidence", f"{confidence * 100:.2f}%")

                st.markdown("### Class Probabilities")

                prob_df = pd.DataFrame(
                    list(probabilities.items()),
                    columns=["Class", "Probability"]
                )

                prob_df["Probability (%)"] = prob_df["Probability"] * 100

                st.dataframe(
                    prob_df[["Class", "Probability (%)"]],
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    prob_df.set_index("Class")["Probability (%)"]
                )

                st.markdown("### Grad-CAM Heatmap")

                gradcam_dir = "dashboard_gradcam_outputs"
                os.makedirs(gradcam_dir, exist_ok=True)

                gradcam_path = os.path.join(
                    gradcam_dir,
                    "uploaded_image_gradcam.png"
                )

                generate_gradcam(
                    model=model,
                    image_tensor=image_tensor,
                    target_class=predicted_index,
                    save_path=gradcam_path,
                    device=device,
                    mean=mean,
                    std=std
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        image,
                        caption="Original CT Scan",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        gradcam_path,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )
