# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import io
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from analysis_utils import (
    correlation_table,
    partial_dependence_curve,
    permutation_importance_table,
    standardized_coefficients,
    variance_inflation_factors,
)
from data_utils import excel_sheet_names, numeric_data_frame, read_tabular_data
from metrics_utils import compute_regression_metrics


st.set_page_config(page_title="General Property Predictor 6d2", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #091620;
            --bg-2: #102534;
            --panel: #111f2d;
            --panel-strong: #153047;
            --primary: #6ad3ff;
            --secondary: #5de4c3;
            --text: #eaf8ff;
            --muted: #bfd7e8;
            --border: rgba(106, 211, 255, 0.45);
            --shadow: rgba(93, 228, 195, 0.32);
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
            color: var(--text);
            font-family: "Verdana", "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 1500px;
        }

        h1, h2, h3, h4, h5, h6,
        .stTabs [role="tab"] {
            font-family: "Times New Roman", Georgia, serif;
            letter-spacing: 0.02em;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 31, 45, 0.98), rgba(11, 22, 32, 0.98));
            border-right: 1px solid var(--border);
        }

        .stButton > button {
            border: 0;
            border-radius: 0.75rem;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            color: #072231;
            font-weight: 700;
            box-shadow: 0 8px 22px var(--shadow);
        }

        .stButton > button:hover {
            filter: brightness(1.05);
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 2px rgba(106, 211, 255, 0.7), 0 8px 22px var(--shadow);
        }

        .stDataFrame, .stSelectbox, .stMultiSelect, .stNumberInput, [data-testid="stFileUploader"] section {
            background: rgba(17, 31, 45, 0.9);
            border: 1px solid var(--border);
            border-radius: 0.8rem;
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            background: rgba(17, 31, 45, 0.7);
            border: 1px solid var(--border);
            color: var(--muted);
            border-radius: 0.7rem 0.7rem 0 0;
            padding: 0.5rem 0.9rem;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, rgba(106, 211, 255, 0.15), rgba(93, 228, 195, 0.12));
            border-color: rgba(106, 211, 255, 0.7);
            color: var(--text);
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("General Property Predictor — 6d2")

TREE_MODELS = {
    "Random Forest", "Gradient Boosting", "XGBoost", "CatBoost", "Extra Trees",
}
LINEAR_MODELS = {"Linear Regression", "Ridge Regression"}
MODEL_NAMES = [
    "Linear Regression",
    "Polynomial Regression deg2",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Ridge Regression",
    "KNN",
    "Neural Network",
    "CatBoost",
    "Lasso Regression",
    "Elastic Net",
    "SVR (RBF)",
    "Extra Trees",
    "HistGradientBoosting",
]


def scaled_regressor(estimator):
    """Standardise inputs and target so regularisation strength is unit-free."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()), ("est", estimator)]),
        transformer=StandardScaler(),
    )


def make_models():
    return [
        ("Linear Regression", LinearRegression()),
        (
            "Polynomial Regression deg2",
            Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lin", LinearRegression()),
            ]),
        ),
        ("Random Forest", RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
        ("XGBoost", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("KNN", KNeighborsRegressor(n_neighbors=5, weights="distance")),
        ("Neural Network", TransformedTargetRegressor(
            regressor=Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 32), activation="relu",
                    alpha=0.05, learning_rate_init=0.001,
                    max_iter=2000, random_state=42)),
            ]),
            transformer=StandardScaler(),
        )),
        ("CatBoost", CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0)),
        ("Lasso Regression", scaled_regressor(
            Lasso(alpha=0.01, max_iter=50000))),
        ("Elastic Net", scaled_regressor(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000))),
        ("SVR (RBF)", scaled_regressor(SVR(kernel="rbf", C=10.0, epsilon=0.05))),
        ("Extra Trees", ExtraTreesRegressor(
            n_estimators=300, random_state=42, n_jobs=-1)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, min_samples_leaf=3,
            random_state=42)),
    ]


def standardized_effects(model, feature_names, x_train, y_train):
    """Per-standard-deviation effect on the target for linear-type models."""
    regressor = getattr(model, "regressor_", None)
    if regressor is not None and hasattr(regressor, "named_steps"):
        coefficients = getattr(regressor.named_steps.get("est"), "coef_", None)
        if coefficients is not None:
            return pd.Series(np.ravel(coefficients), index=feature_names)
        return None
    return standardized_coefficients(model, feature_names, x_train, y_train)


def formula_markdown(model, target_name: str, model_name: str, feature_names):
    if model_name == "Polynomial Regression deg2":
        poly = model.named_steps["poly"]
        estimator = model.named_steps["lin"]
        names = poly.get_feature_names_out(feature_names)
    else:
        estimator = model
        names = feature_names

    terms = " ".join(
        f"{coef:+.6f}·{feature}"
        for coef, feature in zip(estimator.coef_, names)
    )
    return (
        f"**{model_name} formula for {target_name}**\n\n"
        f"```\n{target_name} = {estimator.intercept_:.6f} {terms}\n```"
    )


def parity_figure(x_train, x_test, y_train, y_test, model, target_name):
    x_full = pd.concat([x_train, x_test])
    y_full = pd.concat([y_train, y_test])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, (x_values, y_values, subtitle) in zip(axes, [
        (x_test, y_test, "Test data"),
        (x_full, y_full, "Full dataset (train + test)"),
    ]):
        predictions = model.predict(x_values)
        axis.scatter(y_values, predictions, alpha=0.7, edgecolor="k", s=30)
        minimum = min(y_values.min(), predictions.min())
        maximum = max(y_values.max(), predictions.max())
        axis.plot([minimum, maximum], [minimum, maximum], "r--", lw=2)
        axis.set_xlabel(f"True {target_name}")
        axis.set_ylabel(f"Predicted {target_name}")
        axis.set_title(subtitle)
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"{target_name} — Parity Plots", fontsize=13)
    figure.tight_layout()
    return figure


THUMBNAIL_WIDTH = 560


def render_zoomable_figure(figure, label, key_suffix, preview_size=(9, 5), zoom_size=(16, 9)):
    def to_png(size, dpi):
        # Resize in place and restore: deepcopy of a figure is unsupported and
        # fails on categorical axes with recent matplotlib/Python versions.
        original_size = figure.get_size_inches().copy()
        figure.set_size_inches(*size)
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        finally:
            figure.set_size_inches(*original_size)
        return buffer.getvalue()

    # Fixed-width image: st.pyplot would stretch any figure to the container width.
    st.image(to_png(preview_size, 100), width=THUMBNAIL_WIDTH)

    @st.dialog(label, width="large")
    def show_zoom():
        st.image(to_png(zoom_size, 170), width="stretch")

    if st.button(f"⛶ Full screen — {label}", key=f"zoom_open_{key_suffix}"):
        show_zoom()


def preferred_targets(columns):
    return [columns[-1]] if columns else []


@st.cache_resource(show_spinner=False)
def train_all(file_bytes, file_name, sheet_name, feature_names, target_names):
    raw = read_tabular_data(file_bytes, file_name, sheet_name)
    numeric = numeric_data_frame(raw)
    selected = list(feature_names) + list(target_names)
    model_data = numeric[selected].dropna()
    train_frame, test_frame = train_test_split(
        model_data, test_size=0.2, random_state=42
    )
    x_train = train_frame[list(feature_names)]
    x_test = test_frame[list(feature_names)]

    results = {}
    for target_name in target_names:
        y_train = train_frame[target_name]
        y_test = test_frame[target_name]
        models = make_models()
        for _, model in models:
            model.fit(x_train, y_train)
        results[target_name] = {
            "models": models,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    return results


@st.cache_resource(show_spinner=False)
def analyse_target(file_bytes, file_name, sheet_name, feature_names, target_names, target_name):
    result = train_all(file_bytes, file_name, sheet_name, feature_names, target_names)[target_name]
    features = list(feature_names)
    full_x = pd.concat([result["x_train"], result["x_test"]])
    full_y = pd.concat([result["y_train"], result["y_test"]])
    frame = full_x.copy()
    frame[target_name] = full_y.values
    return {
        "association": correlation_table(frame, features, target_name),
        "feature_corr": full_x.corr(method="pearson"),
        "vif": variance_inflation_factors(full_x, features),
        "permutation": permutation_importance_table(
            result["models"], result["x_test"], result["y_test"]
        ),
    }


@st.cache_resource(show_spinner=False)
def explain_model(file_bytes, file_name, sheet_name, feature_names, target_names,
                  target_name, model_name):
    result = train_all(file_bytes, file_name, sheet_name, feature_names, target_names)[target_name]
    model = dict(result["models"])[model_name]
    features = list(feature_names)
    x_train = result["x_train"]
    data = pd.concat([x_train, result["x_test"]])
    background = shap.sample(x_train, min(50, len(x_train)), random_state=42)

    if model_name in TREE_MODELS:
        data = data.sample(min(300, len(data)), random_state=42)
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(data))
        base = float(np.ravel(explainer.expected_value)[0])
        explanation = shap.Explanation(
            values=values, base_values=np.full(len(data), base),
            data=data.values, feature_names=features,
        )
        method = "TreeExplainer (exact)"
    elif model_name in LINEAR_MODELS:
        explanation = shap.LinearExplainer(model, background)(data)
        method = "LinearExplainer (exact)"
    else:
        data = data.sample(min(60, len(data)), random_state=42)
        explanation = shap.Explainer(model.predict, background)(data)
        method = "Permutation explainer (model-agnostic, 60-row sample)"
    return explanation, method


for key, value in [("training_config", None), ("trained", False)]:
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Upload and Train")
    uploaded_file = st.file_uploader(
        "CSV or Excel file", type=["csv", "xlsx", "xlsm"]
    )
    if uploaded_file is None:
        st.info("Upload a CSV, XLSX, or XLSM file to choose variables.")
        st.stop()

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    sheet_name = None

    try:
        if file_name.lower().endswith((".xlsx", ".xlsm")):
            sheet_name = st.selectbox(
                "Worksheet", excel_sheet_names(file_bytes)
            )
        raw_frame = read_tabular_data(file_bytes, file_name, sheet_name)
    except (
        InvalidFileException,
        ImportError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()

    numeric_frame = numeric_data_frame(raw_frame)
    numeric_columns = list(numeric_frame.columns)
    st.caption("Preview (first 10 rows)")
    st.dataframe(raw_frame.head(10), hide_index=True, use_container_width=True)

    if len(numeric_columns) < 2:
        st.error("The selected sheet needs at least two numeric columns.")
        st.stop()

    default_targets = preferred_targets(numeric_columns)
    default_features = [
        column for column in numeric_columns if column not in default_targets
    ]
    st.caption(
        "Each row should be one observation and the header row should contain "
        "variable names. Non-numeric columns are excluded."
    )
    feature_selection = st.multiselect(
        "Input variables",
        options=numeric_columns,
        default=default_features,
        help="Choose one or more values the model will use to make predictions.",
    )
    target_selection = st.multiselect(
        "Target variables to predict",
        options=numeric_columns,
        default=default_targets,
        help="The app trains a separate set of fourteen models for each target.",
    )

    overlap = sorted(set(feature_selection) & set(target_selection))
    selected_columns = feature_selection + target_selection
    if overlap:
        st.error(
            "A column cannot be both an input and a target: "
            + ", ".join(overlap)
        )
    elif not feature_selection or not target_selection:
        st.warning("Select at least one input and one target variable.")
    else:
        usable_rows = numeric_frame[selected_columns].dropna().shape[0]
        st.caption(
            f"{usable_rows} rows have numeric values in all selected columns."
        )
        if usable_rows < 10:
            st.warning("At least 10 complete rows are required to train models.")

    train_button = st.button(
        "Train Models",
        type="primary",
        disabled=(
            not feature_selection
            or not target_selection
            or bool(overlap)
            or numeric_frame[selected_columns].dropna().shape[0] < 10
        ),
        use_container_width=True,
    )

    config = (
        hashlib.sha256(file_bytes).hexdigest(),
        file_name,
        sheet_name,
        tuple(feature_selection),
        tuple(target_selection),
    )
    if train_button:
        st.session_state.training_config = config
        st.session_state.trained = True

if (
    not st.session_state.trained
    or st.session_state.training_config != config
):
    st.info(
        "Choose input and target variables in the sidebar, then click "
        "**Train Models**."
    )
    st.stop()

feature_cols = list(feature_selection)
target_names = list(target_selection)
training_data = numeric_frame[feature_cols + target_names].dropna()

with st.spinner("Training models for the selected targets…"):
    target_results = train_all(
        file_bytes,
        file_name,
        sheet_name,
        tuple(feature_cols),
        tuple(target_names),
    )

st.success(
    f"Models trained · Targets: **{', '.join(target_names)}** · "
    f"Inputs: **{', '.join(feature_cols)}**"
)

tab_names = ["Feature Analysis", "Model Comparison", "Predict"] + MODEL_NAMES
tabs = st.tabs(tab_names)

with tabs[0]:
    for target_name in target_names:
        result = target_results[target_name]
        models = result["models"]
        x_train = result["x_train"]
        x_test = result["x_test"]
        y_train = result["y_train"]
        y_test = result["y_test"]
        full_frame = pd.concat([x_train, x_test])
        full_frame[target_name] = pd.concat([
            result["y_train"], result["y_test"]
        ]).values

        analysis = analyse_target(
            file_bytes, file_name, sheet_name,
            tuple(feature_cols), tuple(target_names), target_name,
        )

        st.header(f"Analysis: {target_name}")
        st.subheader("1. Correlation and association (inputs → target)")
        st.caption(
            "Pearson: linear. Spearman and Kendall: monotonic (rank-based, "
            "Kendall is more robust on small samples). Distance correlation: "
            "any dependence, 0 = independent. Mutual information: any "
            "dependence, in nats (unbounded)."
        )
        association = analysis["association"]
        signed = association[["Pearson", "Spearman", "Kendall"]].T
        unsigned = association[["Distance corr.", "Mutual info"]]
        figure, (heat_axis, bar_axis) = plt.subplots(
            1, 2, figsize=(max(11, 2.6 * len(feature_cols)), 3.2),
            gridspec_kw={"width_ratios": [1.2, 1]},
        )
        image = heat_axis.imshow(
            signed.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto"
        )
        heat_axis.set_xticks(range(len(feature_cols)))
        heat_axis.set_xticklabels(feature_cols, rotation=30, ha="right")
        heat_axis.set_yticks(range(len(signed.index)))
        heat_axis.set_yticklabels(signed.index)
        for row_index in range(signed.shape[0]):
            for column_index in range(signed.shape[1]):
                value = signed.values[row_index, column_index]
                heat_axis.text(
                    column_index, row_index, f"{value:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(value) > 0.5 else "black",
                )
        heat_axis.set_title("Signed correlation")
        figure.colorbar(image, ax=heat_axis, fraction=0.046)
        unsigned.plot(kind="bar", ax=bar_axis)
        bar_axis.set_title("Dependence strength")
        bar_axis.grid(axis="y", alpha=0.3)
        bar_axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        render_zoomable_figure(
            figure,
            f"{target_name} correlation",
            f"corr_{target_name}",
            preview_size=(11, 4),
            zoom_size=(17, 6),
        )
        plt.close(figure)
        st.dataframe(association.round(4), use_container_width=True)

        if len(feature_cols) > 1:
            st.markdown("**Input-to-input correlation and multicollinearity**")
            st.caption(
                "Highly correlated inputs share credit in importance measures. "
                "VIF above about 5–10 signals strong multicollinearity."
            )
            feature_corr = analysis["feature_corr"]
            figure, axis = plt.subplots(
                figsize=(max(5, 0.9 * len(feature_cols) + 3), max(4, 0.8 * len(feature_cols) + 2))
            )
            image = axis.imshow(feature_corr.values, cmap="RdBu", vmin=-1, vmax=1)
            axis.set_xticks(range(len(feature_cols)))
            axis.set_xticklabels(feature_cols, rotation=45, ha="right")
            axis.set_yticks(range(len(feature_cols)))
            axis.set_yticklabels(feature_cols)
            for row_index in range(len(feature_cols)):
                for column_index in range(len(feature_cols)):
                    value = feature_corr.values[row_index, column_index]
                    axis.text(
                        column_index, row_index, f"{value:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(value) > 0.5 else "black",
                    )
            figure.colorbar(image, ax=axis, fraction=0.046)
            figure.tight_layout()
            render_zoomable_figure(
                figure,
                f"{target_name} input correlation",
                f"featcorr_{target_name}",
                preview_size=(6, 5),
                zoom_size=(10, 8),
            )
            plt.close(figure)
            st.dataframe(
                analysis["vif"].round(3).to_frame("VIF"), use_container_width=True
            )
        plt.close("all")

        st.subheader("2. Feature Importances")
        permutation = analysis["permutation"]
        positive = permutation.clip(lower=0)
        totals = positive.sum(axis=0).replace(0, np.nan)
        consensus = (positive / totals).mean(axis=1).sort_values(ascending=False)

        st.markdown("**2a. Consensus importance (all models)**")
        st.caption(
            "Permutation importance is the drop in test R² when one input is "
            "shuffled. Each model's scores are normalised to sum to 1 and "
            "averaged across all models. Test sets are small, so treat "
            "differences between close features as noise."
        )
        figure, axis = plt.subplots(figsize=(max(7, 1.8 * len(feature_cols)), 3.8))
        consensus.plot(kind="bar", ax=axis, color="#2a9df4")
        axis.set_ylabel("Mean normalised importance")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        render_zoomable_figure(
            figure, f"{target_name} consensus importance",
            f"consensus_{target_name}", preview_size=(9, 4.5), zoom_size=(15, 7.5),
        )
        plt.close(figure)

        st.markdown("**2b. Permutation importance by model**")
        figure, axis = plt.subplots(
            figsize=(max(7, 1.4 * len(feature_cols) + 3), 0.45 * len(MODEL_NAMES) + 2)
        )
        image = axis.imshow(permutation.T.values, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(feature_cols)))
        axis.set_xticklabels(feature_cols, rotation=30, ha="right")
        axis.set_yticks(range(len(permutation.columns)))
        axis.set_yticklabels(permutation.columns)
        top = np.nanmax(permutation.values) or 1.0
        for row_index in range(permutation.shape[1]):
            for column_index in range(permutation.shape[0]):
                value = permutation.values[column_index, row_index]
                axis.text(
                    column_index, row_index, f"{value:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if value > 0.6 * top else "white",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, label="R² drop")
        figure.tight_layout()
        render_zoomable_figure(
            figure, f"{target_name} permutation importance",
            f"permutation_{target_name}", preview_size=(9, 6), zoom_size=(15, 9),
        )
        plt.close(figure)

        st.markdown("**2c. Built-in tree importances**")
        importance_rows = []
        for model_name, model in models:
            if model_name in TREE_MODELS:
                importance_rows.append(
                    pd.Series(
                        model.feature_importances_,
                        index=feature_cols,
                        name=model_name,
                    )
                )
        importance_frame = pd.DataFrame(importance_rows)
        figure, axis = plt.subplots(
            figsize=(max(8, 2.5 * len(feature_cols)), 4)
        )
        importance_frame.T.plot(kind="bar", ax=axis)
        axis.set_ylabel("Feature Importance")
        axis.set_title(f"Tree-model feature importances — {target_name}")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        render_zoomable_figure(
            figure,
            f"{target_name} importances",
            f"importance_{target_name}",
            preview_size=(9, 4.5),
            zoom_size=(15, 7.5),
        )
        plt.close(figure)

        st.markdown("**2d. Standardised coefficients (linear-type models)**")
        st.caption(
            "Change in the target, in target standard deviations, per one "
            "standard deviation increase of the input."
        )
        effects = {}
        for model_name, model in models:
            if model_name in LINEAR_MODELS or model_name in {"Lasso Regression", "Elastic Net"}:
                series = standardized_effects(model, feature_cols, x_train, y_train)
                if series is not None:
                    effects[model_name] = series
        if effects:
            figure, axis = plt.subplots(figsize=(max(8, 2.5 * len(feature_cols)), 4))
            pd.DataFrame(effects).plot(kind="bar", ax=axis)
            axis.axhline(0, color="black", lw=0.8)
            axis.set_ylabel("Standardised coefficient")
            axis.grid(axis="y", alpha=0.3)
            figure.tight_layout()
            render_zoomable_figure(
                figure, f"{target_name} standardised coefficients",
                f"stdcoef_{target_name}", preview_size=(9, 4.5), zoom_size=(15, 7.5),
            )
            plt.close(figure)
            st.caption(
                "Lasso and Elastic Net set unhelpful inputs exactly to zero."
            )

        test_scores = {}
        for model_name, model in models:
            score = compute_regression_metrics(y_test, model.predict(x_test))["r2"]
            test_scores[model_name] = score if np.isfinite(score) else -math.inf
        best_name = max(test_scores, key=test_scores.get)
        explain_name = st.selectbox(
            f"Model to explain for {target_name} (SHAP and partial dependence)",
            MODEL_NAMES,
            index=MODEL_NAMES.index(best_name),
            key=f"explain_model_{target_name}",
        )
        st.caption(f"Default is the best model by test R²: {best_name}")

        st.subheader(f"3. SHAP values — {explain_name}")
        with st.spinner("Computing SHAP values…"):
            explanation, shap_method = explain_model(
                file_bytes, file_name, sheet_name,
                tuple(feature_cols), tuple(target_names), target_name, explain_name,
            )
        st.caption(f"{shap_method}; {len(explanation.values)} rows explained.")
        mean_abs = pd.Series(
            np.abs(explanation.values).mean(axis=0), index=feature_cols
        ).sort_values(ascending=False)

        shap.plots.bar(explanation, show=False)
        render_zoomable_figure(
            plt.gcf(), f"{target_name} SHAP bar",
            f"shap_{target_name}_bar", preview_size=(9, 4.8), zoom_size=(15, 8),
        )
        plt.close("all")
        shap.plots.beeswarm(explanation, show=False)
        render_zoomable_figure(
            plt.gcf(), f"{target_name} SHAP beeswarm",
            f"shap_{target_name}_dot", preview_size=(9, 4.8), zoom_size=(15, 8),
        )
        plt.close("all")

        dependence_feature = st.selectbox(
            "SHAP dependence feature", list(mean_abs.index),
            key=f"shap_dep_{target_name}",
        )
        shap.plots.scatter(
            explanation[:, dependence_feature], color=explanation, show=False
        )
        render_zoomable_figure(
            plt.gcf(), f"{target_name} SHAP dependence",
            f"shap_{target_name}_dependence", preview_size=(8, 4.8), zoom_size=(14, 8),
        )
        plt.close("all")

        row_number = st.number_input(
            "Explain a single row (waterfall)", min_value=0,
            max_value=len(explanation.values) - 1, value=0, step=1,
            key=f"shap_row_{target_name}",
        )
        shap.plots.waterfall(explanation[int(row_number)], show=False)
        render_zoomable_figure(
            plt.gcf(), f"{target_name} SHAP waterfall",
            f"shap_{target_name}_waterfall", preview_size=(9, 4.8), zoom_size=(15, 8),
        )
        plt.close("all")

        st.subheader(f"4. Partial dependence — {explain_name}")
        st.caption(
            "Average predicted value while one input sweeps its observed "
            "range and the others keep their observed values."
        )
        explained_model = dict(models)[explain_name]
        columns_count = min(len(feature_cols), 3)
        rows_count = math.ceil(len(feature_cols) / columns_count)
        figure, axes = plt.subplots(
            rows_count, columns_count,
            figsize=(4.2 * columns_count, 3.2 * rows_count), squeeze=False,
        )
        pd_frame = pd.concat([x_train, x_test])
        for index, feature_name in enumerate(feature_cols):
            axis = axes[index // columns_count][index % columns_count]
            grid, curve = partial_dependence_curve(
                explained_model, pd_frame, feature_name
            )
            axis.plot(grid, curve, marker="o", ms=3)
            axis.set_xlabel(feature_name)
            axis.set_ylabel(f"Mean {target_name}")
            axis.grid(alpha=0.3)
        for index in range(len(feature_cols), rows_count * columns_count):
            axes[index // columns_count][index % columns_count].axis("off")
        figure.tight_layout()
        render_zoomable_figure(
            figure, f"{target_name} partial dependence",
            f"pdp_{target_name}", preview_size=(9, 3.2 * rows_count / 1.2),
            zoom_size=(15, 5.5 * rows_count / 1.2),
        )
        plt.close(figure)

with tabs[1]:
    st.subheader("Model comparison — held-out test set (80/20 split)")
    for target_name in target_names:
        result = target_results[target_name]
        comparison_rows = []
        for model_name, model in result["models"]:
            test_metrics = compute_regression_metrics(
                result["y_test"], model.predict(result["x_test"])
            )
            comparison_rows.append({
                "Model": model_name,
                "R² (test)": test_metrics["r2"],
                "RMSE": test_metrics["rmse"],
                "MAE": test_metrics["mae"],
                "MAPE (%)": test_metrics["mape"],
            })
        comparison = pd.DataFrame(comparison_rows).sort_values(
            "R² (test)", ascending=False, na_position="last"
        )
        st.markdown(f"**{target_name}**")
        st.dataframe(comparison.round(6), hide_index=True, use_container_width=True)
        figure, axis = plt.subplots(figsize=(9, 0.4 * len(comparison) + 1.5))
        scores = comparison["R² (test)"].fillna(0).clip(lower=-1)
        axis.barh(comparison["Model"], scores, color="#2a9df4")
        axis.invert_yaxis()
        axis.axvline(0, color="black", lw=0.8)
        axis.set_xlabel("Test R² (clipped at -1)")
        axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()
        render_zoomable_figure(
            figure, f"{target_name} model comparison",
            f"comparison_{target_name}", preview_size=(9, 0.4 * len(comparison) + 1.5),
            zoom_size=(15, 0.6 * len(comparison) + 2),
        )
        plt.close(figure)

with tabs[2]:
    st.subheader("Predict target values for a new observation")
    prediction_columns = st.columns(min(len(feature_cols), 4))
    prediction_values = {}
    for index, feature_name in enumerate(feature_cols):
        with prediction_columns[index % len(prediction_columns)]:
            prediction_values[feature_name] = st.number_input(
                feature_name,
                value=float(training_data[feature_name].median()),
                format="%.6g",
                key=f"prediction_{feature_name}",
            )
    if st.button("Predict", type="primary"):
        new_data = pd.DataFrame([prediction_values], columns=feature_cols)
        prediction_rows = []
        for model_index, model_name in enumerate(MODEL_NAMES):
            row = {"Model": model_name}
            for target_name in target_names:
                model = target_results[target_name]["models"][model_index][1]
                row[target_name] = float(model.predict(new_data)[0])
            prediction_rows.append(row)
        st.dataframe(
            pd.DataFrame(prediction_rows).round(6),
            hide_index=True,
            use_container_width=True,
        )

for tab, model_name in zip(tabs[3:], MODEL_NAMES):
    with tab:
        st.subheader("Metrics — held-out test set (80/20 split)")
        metric_rows = []
        for target_name in target_names:
            result = target_results[target_name]
            model = dict(result["models"])[model_name]
            y_test = result["y_test"]
            y_test_pred = model.predict(result["x_test"])
            y_full = pd.concat([result["y_train"], y_test])
            x_full = pd.concat([result["x_train"], result["x_test"]])
            full_metrics = compute_regression_metrics(
                y_full, model.predict(x_full)
            )
            test_metrics = compute_regression_metrics(y_test, y_test_pred)
            metric_rows.append({
                "Target": target_name,
                "R² (test)": f"{test_metrics['r2']:.4f}",
                "R² (all)": f"{full_metrics['r2']:.4f}",
                "MAE": f"{test_metrics['mae']:.6f}",
                "RMSE": f"{test_metrics['rmse']:.6f}",
                "MAPE (%) test": f"{test_metrics['mape']:.6f}",
                "MAPE (%) all": f"{full_metrics['mape']:.6f}",
                "AAD (%) test": f"{test_metrics['aad']:.6f}",
                "AAD (%) all": f"{full_metrics['aad']:.6f}",
                "MAE% of Mean test": f"{test_metrics['mae_pct_of_mean']:.6f}",
                "MAE% of Mean all": f"{full_metrics['mae_pct_of_mean']:.6f}",
            })
        st.dataframe(
            pd.DataFrame(metric_rows), hide_index=True, use_container_width=True
        )

        if model_name in {
            "Linear Regression", "Polynomial Regression deg2", "Ridge Regression"
        }:
            st.subheader("Fitted formulas")
            for target_name in target_names:
                model = dict(target_results[target_name]["models"])[model_name]
                st.markdown(
                    formula_markdown(
                        model, target_name, model_name, feature_cols
                    )
                )

        if model_name in {"Lasso Regression", "Elastic Net"}:
            st.subheader("Standardised coefficients")
            st.caption(
                "Inputs and target are standardised; zero means the input was "
                "dropped by the L1 penalty."
            )
            coefficient_rows = {}
            for target_name in target_names:
                result = target_results[target_name]
                model = dict(result["models"])[model_name]
                coefficient_rows[target_name] = standardized_effects(
                    model, feature_cols, result["x_train"], result["y_train"]
                )
            st.dataframe(pd.DataFrame(coefficient_rows).round(6), use_container_width=True)

        st.subheader("Parity Plots")
        for target_name in target_names:
            result = target_results[target_name]
            model = dict(result["models"])[model_name]
            figure = parity_figure(
                result["x_train"],
                result["x_test"],
                result["y_train"],
                result["y_test"],
                model,
                target_name,
            )
            render_zoomable_figure(
                figure,
                f"{model_name} {target_name}",
                f"parity_{model_name}_{target_name}",
                preview_size=(9, 4.8),
                zoom_size=(15, 8),
            )
            plt.close(figure)
