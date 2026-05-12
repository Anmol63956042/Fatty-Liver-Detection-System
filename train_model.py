"""
train_model.py
==============
Training pipeline for AI-Powered Fatty Liver Disease Detection System.

Workflow:
    1. Data loading and schema standardisation
    2. Exploratory Data Analysis (EDA) with visualisations
    3. Data cleaning and preprocessing
    4. Model training: Logistic Regression, Decision Tree, Random Forest, ANN
    5. Model evaluation and comparative scoring plots
    6. Serialisation of trained artefacts for inference

Output artefacts (models/):
    - scaler.pkl
    - label_encoder_gender.pkl
    - feature_cols.pkl
    - random_forest_model.pkl
    - ann_model.keras

Output plots (plots/):
    - eda_class_distribution.png
    - eda_correlation_heatmap.png
    - eda_feature_distributions.png
    - eda_boxplots_by_class.png
    - eda_age_distribution.png
    - model_scoring_comparison.png
    - confusion_matrix_rf.png
    - feature_importance_rf.png
    - ann_training_curve.png
"""

# ---------------------------------------------------------------------------
# Standard library and third-party imports
# ---------------------------------------------------------------------------
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
ANN_EPOCHS    = 100
ANN_BATCH     = 32
ANN_PATIENCE  = 10
PLOTS_DIR     = "plots"
MODELS_DIR    = "models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Seaborn aesthetic settings for publication-quality figures
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = {"Disease": "#e74c3c", "No Disease": "#2ecc71"}


# ---------------------------------------------------------------------------
# Section 1: Data Loading
# ---------------------------------------------------------------------------

def locate_dataset(filename: str) -> str:
    """
    Resolve the file path by searching common locations relative to the
    working directory.  Raises FileNotFoundError with a descriptive message
    if the file cannot be found.
    """
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Dataset '{filename}' not found. "
        f"Place it in the project root or inside a 'data/' subfolder."
    )


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load CSV with latin-1 encoding to handle non-UTF-8 byte sequences
    present in the Indian Liver Patient Dataset export.
    """
    path = locate_dataset(filename)
    df   = pd.read_csv(path, encoding="latin-1")
    return df


# ---------------------------------------------------------------------------
# Section 2: Schema Standardisation
# ---------------------------------------------------------------------------

RENAME_MAP = {
    # Verbose original column names -> clean internal names
    "Age of the patient":                    "Age",
    "Gender of the patient":                 "Gender",
    "Total Bilirubin":                       "Total_Bilirubin",
    "Direct Bilirubin":                      "Direct_Bilirubin",
    "Alkphos Alkaline Phosphotase":          "Alkaline_Phosphotase",
    "Sgpt Alamine Aminotransferase":         "Alamine_Aminotransferase",
    "Sgot Aspartate Aminotransferase":       "Aspartate_Aminotransferase",
    "Total Protiens":                        "Total_Protiens",
    "ALB Albumin":                           "Albumin",
    "A/G Ratio Albumin and Globulin Ratio":  "Albumin_Globulin_Ratio",
    "Result":                                "Dataset",
    # Pass-through mappings for already-clean column names
    "Age":                        "Age",
    "Gender":                     "Gender",
    "Total_Bilirubin":            "Total_Bilirubin",
    "Direct_Bilirubin":           "Direct_Bilirubin",
    "Alkaline_Phosphotase":       "Alkaline_Phosphotase",
    "Alamine_Aminotransferase":   "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase": "Aspartate_Aminotransferase",
    "Total_Protiens":             "Total_Protiens",
    "Albumin":                    "Albumin",
    "Albumin_and_Globulin_Ratio": "Albumin_Globulin_Ratio",
    "Dataset":                    "Dataset",
}

FEATURE_DISPLAY = {
    "Age":                       "Age",
    "Gender":                    "Gender",
    "Total_Bilirubin":           "Total Bilirubin",
    "Direct_Bilirubin":          "Direct Bilirubin",
    "Alkaline_Phosphotase":      "Alkaline Phosphotase",
    "Alamine_Aminotransferase":  "SGPT (Alamine AT)",
    "Aspartate_Aminotransferase":"SGOT (Aspartate AT)",
    "Total_Protiens":            "Total Proteins",
    "Albumin":                   "Albumin (ALB)",
    "Albumin_Globulin_Ratio":    "A/G Ratio",
}


def standardise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all column names and apply the
    rename mapping to produce consistent internal column identifiers.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns},
              inplace=True)
    return df


# ---------------------------------------------------------------------------
# Section 3: Exploratory Data Analysis
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, target_col: str = "Dataset") -> None:
    """
    Generate and persist five EDA visualisations covering class balance,
    inter-feature correlations, univariate distributions, engine biomarkers
    by class, and age demographics.

    Parameters
    ----------
    df         : DataFrame after schema standardisation but before encoding.
    target_col : Name of the binary target column ('Dataset').
    """

    # Derive a human-readable class label column for plotting
    df_plot = df.copy()
    df_plot["Class"] = df_plot[target_col].apply(
        lambda x: "Disease" if int(x) == 1 else "No Disease"
    )

    # -- EDA 1: Class Distribution -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    class_counts = df_plot["Class"].value_counts()
    axes[0].bar(class_counts.index, class_counts.values,
                color=["#e74c3c", "#2ecc71"], edgecolor="black", width=0.5)
    axes[0].set_title("Class Distribution (Count)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of Patients")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    axes[1].pie(class_counts.values,
                labels=class_counts.index,
                colors=["#e74c3c", "#2ecc71"],
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Distribution (Proportion)")

    fig.suptitle("Target Class Balance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_class_distribution.png"), dpi=150)
    plt.close()

    # -- EDA 2: Correlation Heatmap ------------------------------------------
    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix  = df_plot[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix – All Numeric Features",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_correlation_heatmap.png"), dpi=150)
    plt.close()

    # -- EDA 3: Feature Distributions (KDE) ----------------------------------
    plot_features = [
        "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Albumin_Globulin_Ratio",
    ]
    plot_features = [f for f in plot_features if f in df_plot.columns]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(plot_features):
        for cls, grp in df_plot.groupby("Class"):
            axes[i].hist(grp[feat].dropna(), bins=30, alpha=0.5,
                         label=cls, color=PALETTE[cls], edgecolor="none",
                         density=True)
            grp[feat].dropna().plot.kde(ax=axes[i], color=PALETTE[cls],
                                        linewidth=1.8)
        axes[i].set_title(FEATURE_DISPLAY.get(feat, feat))
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
    fig.suptitle("Feature Density Distributions by Disease Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_feature_distributions.png"), dpi=150)
    plt.close()

    # -- EDA 4: Box Plots – Liver Enzyme Biomarkers by Class -----------------
    enzyme_features = [
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Bilirubin",
    ]
    enzyme_features = [f for f in enzyme_features if f in df_plot.columns]

    fig, axes = plt.subplots(1, len(enzyme_features),
                             figsize=(4 * len(enzyme_features), 5))
    for ax, feat in zip(axes, enzyme_features):
        sns.boxplot(data=df_plot, x="Class", y=feat, ax=ax,
                    palette=PALETTE, flierprops={"markersize": 3})
        ax.set_title(FEATURE_DISPLAY.get(feat, feat))
        ax.set_xlabel("")
    fig.suptitle("Liver Enzyme Levels by Disease Class (Outliers Visible)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_boxplots_by_class.png"), dpi=150)
    plt.close()

    # -- EDA 5: Age Demographics ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(data=df_plot, x="Age", hue="Class",
                 palette=PALETTE, bins=20, kde=True,
                 multiple="layer", ax=axes[0], alpha=0.6)
    axes[0].set_title("Age Distribution by Disease Class")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")

    age_bins   = pd.cut(df_plot["Age"], bins=[0, 20, 35, 50, 65, 120],
                        labels=["<20", "20-35", "35-50", "50-65", "65+"])
    age_class  = pd.crosstab(age_bins, df_plot["Class"], normalize="index") * 100
    age_class.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"],
                   edgecolor="black", width=0.6)
    axes[1].set_title("Disease Prevalence (%) by Age Group")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Class")
    axes[1].tick_params(axis="x", rotation=0)

    fig.suptitle("Patient Age Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_age_distribution.png"), dpi=150)
    plt.close()

    print(f"EDA plots saved to '{PLOTS_DIR}/'")


# ---------------------------------------------------------------------------
# Section 4: Data Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, target_col: str = "Dataset"):
    """
    Apply full preprocessing pipeline:
      - Encode the binary Gender categorical variable
      - Impute remaining missing values with column-wise median
      - Binarise the target label (1 = disease, 0 = no disease)
      - Stratified train/test split
      - Z-score normalisation via StandardScaler

    Returns
    -------
    X_train_sc, X_test_sc, y_train, y_test : numpy arrays
    scaler                                 : fitted StandardScaler
    le                                     : fitted LabelEncoder for Gender
    feature_cols                           : ordered list of feature names
    """

    # Encode gender; fill NaN gender values before encoding
    le = LabelEncoder()
    df["Gender"] = df["Gender"].fillna("Male").astype(str).str.strip()
    df["Gender"] = le.fit_transform(df["Gender"])
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_gender.pkl"))

    # Median imputation for all remaining numeric missing values
    missing_before = df.isnull().sum().sum()
    df.fillna(df.median(numeric_only=True), inplace=True)
    missing_after  = df.isnull().sum().sum()
    print(f"Missing values imputed: {missing_before} -> {missing_after}")

    # Binarise target: original label 1 = disease present, 2 = no disease
    df[target_col] = df[target_col].apply(lambda x: 1 if int(x) == 1 else 0)

    feature_cols = [c for c in df.columns if c != target_col]
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    return X_train_sc, X_test_sc, y_train, y_test, scaler, le, feature_cols


# ---------------------------------------------------------------------------
# Section 5: Model Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train):
    """
    Fit a regularised Logistic Regression classifier.
    max_iter set to 1000 to ensure convergence on scaled medical data.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """
    Fit a Decision Tree with depth constraint to prevent overfitting.
    max_depth=8 balances expressiveness and generalisation.
    """
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Fit a Random Forest ensemble classifier.
    200 trees with max_depth=10 provides robust generalisation.
    Parallelised across all available CPU cores (n_jobs=-1).
    This is the primary model selected for deployment.
    """
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    return model


def train_ann(X_train, y_train, input_dim: int):
    """
    Build and train a feed-forward Artificial Neural Network.

    Architecture:
        Input  -> Dense(128, ReLU) -> BN -> Dropout(0.3)
               -> Dense(64,  ReLU) -> BN -> Dropout(0.2)
               -> Dense(32,  ReLU)
               -> Dense(1,  Sigmoid)

    Training:
        Optimiser : Adam
        Loss      : Binary cross-entropy
        Callback  : EarlyStopping (patience=10) on validation loss
    """
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    es = EarlyStopping(
        monitor="val_loss", patience=ANN_PATIENCE,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=ANN_EPOCHS,
        batch_size=ANN_BATCH,
        validation_split=0.15,
        callbacks=[es],
        verbose=0
    )
    model.save(os.path.join(MODELS_DIR, "ann_model.keras"))
    return model, history


# ---------------------------------------------------------------------------
# Section 6: Evaluation Utilities
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    """Return a dictionary of standard binary classification metrics."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def print_results_table(results: dict) -> None:
    """Print a formatted comparison table of model evaluation metrics."""
    header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>9} {'Recall':>7} {'F1-Score':>8}"
    print("" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>9} {m['precision']:>9} "
              f"{m['recall']:>7} {m['f1']:>8}")
    print("=" * len(header) + "")


# ---------------------------------------------------------------------------
# Section 7: Visualisation – Evaluation Plots
# ---------------------------------------------------------------------------

def plot_scoring_comparison(results: dict) -> None:
    """
    Generate a grouped bar chart comparing Accuracy, Precision, Recall,
    and F1-Score across all trained models.
    Saved to plots/model_scoring_comparison.png.
    """
    models   = list(results.keys())
    metrics  = ["accuracy", "precision", "recall", "f1"]
    labels   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colours  = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    x     = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=colour, edgecolor="black", alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison – All Evaluation Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_scoring_comparison.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred_rf) -> None:
    """
    Plot and save the confusion matrix for the Random Forest model.
    Saved to plots/confusion_matrix_rf.png.
    """
    cm = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                linewidths=0.5, ax=ax)
    ax.set_title("Random Forest - Confusion Matrix", fontweight="bold")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_rf.png"), dpi=150)
    plt.close()


def plot_feature_importance(rf_model, feature_cols: list) -> None:
    """
    Plot Random Forest feature importances (mean decrease in impurity).
    Saved to plots/feature_importance_rf.png.
    """
    fi = pd.Series(rf_model.feature_importances_,
                   index=[FEATURE_DISPLAY.get(c, c) for c in feature_cols]
                   ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", color="steelblue", edgecolor="black", ax=ax)
    ax.set_title("Feature Importance - Random Forest (Mean Decrease Impurity)",
                 fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance_rf.png"), dpi=150)
    plt.close()


def plot_ann_training_curve(history) -> None:
    """
    Plot ANN training and validation accuracy/loss curves over epochs.
    Saved to plots/ann_training_curve.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",
                 color="#3498db", linewidth=1.8)
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy",
                 color="#e74c3c", linewidth=1.8, linestyle="--")
    axes[0].set_title("ANN - Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss",
                 color="#3498db", linewidth=1.8)
    axes[1].plot(history.history["val_loss"], label="Validation Loss",
                 color="#e74c3c", linewidth=1.8, linestyle="--")
    axes[1].set_title("ANN - Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].legend()

    fig.suptitle("Artificial Neural Network Training History",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ann_training_curve.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Section 8: Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- 8.1 Load raw data --------------------------------------------------
    df_raw = load_raw_data("train.csv")

    # ---- 8.2 Standardise column schema --------------------------------------
    df = standardise_schema(df_raw)

    # ---- 8.3 Exploratory Data Analysis (pre-encoding, uses original labels) -
    run_eda(df.copy(), target_col="Dataset")

    # ---- 8.4 Preprocessing --------------------------------------------------
    (X_train_sc, X_test_sc,
     y_train, y_test,
     scaler, le, feature_cols) = preprocess(df, target_col="Dataset")

    # ---- 8.5 Train all models -----------------------------------------------
    print("Training Logistic Regression ...")
    lr    = train_logistic_regression(X_train_sc, y_train)

    print("Training Decision Tree ...")
    dt    = train_decision_tree(X_train_sc, y_train)

    print("Training Random Forest ...")
    rf    = train_random_forest(X_train_sc, y_train)

    print("Training ANN (Neural Network) ...")
    ann, history = train_ann(X_train_sc, y_train,
                             input_dim=X_train_sc.shape[1])

    # ---- 8.6 Evaluate -------------------------------------------------------
    results = {
        "Logistic Regression": compute_metrics(y_test, lr.predict(X_test_sc)),
        "Decision Tree":       compute_metrics(y_test, dt.predict(X_test_sc)),
        "Random Forest":       compute_metrics(y_test, rf.predict(X_test_sc)),
        "ANN": compute_metrics(
            y_test,
            (ann.predict(X_test_sc, verbose=0).flatten() >= 0.5).astype(int)
        ),
    }

    print_results_table(results)

    # Detailed report for the primary model
    y_pred_rf = rf.predict(X_test_sc)
    print("Random Forest - Classification Report:")
    print(classification_report(y_test, y_pred_rf,
                                 target_names=["No Disease", "Disease"]))

    # ---- 8.7 Generate evaluation plots --------------------------------------
    plot_scoring_comparison(results)
    plot_confusion_matrix(y_test, y_pred_rf)
    plot_feature_importance(rf, feature_cols)
    plot_ann_training_curve(history)

    print(f"All plots saved to '{PLOTS_DIR}/'")
    print(f"All model artefacts saved to '{MODELS_DIR}/'")
    print("Training pipeline complete.")