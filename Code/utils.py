"""
Shared helpers for the Credit Card Fraud Detection project.

Every notebook imports this file so that all models are trained and tested on
exactly the same data, with the same random seed, and measured with the same
metrics. Without this, a comparison between models would not be meaningful.

Course : Machine Learning
Student: Lakshay (AP24110010677)
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
)

# ---------------------------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------------------------
# As instructed, the random seed is the last two digits of my university roll
# number AP24110010677, i.e. 77. This same value is used for the train/test
# split, for every model that has internal randomness, and for every subsample,
# so that all results in this project are exactly reproducible.
SEED = 77

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Dataset" / "creditcard.csv"
FIG_DIR = ROOT / "Results" / "figures"
TAB_DIR = ROOT / "Results" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.30
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
RED = "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de", "#fcfcfb"
SERIES = [BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN, VIOLET]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "font.family": "DejaVu Sans", "lines.linewidth": 2.0, "figure.dpi": 120,
})


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save_fig(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"saved figure -> Results/figures/{name}")
    return p


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(verbose=True):
    """Load the ULB credit card dataset and remove exact duplicate rows."""
    df = pd.read_csv(DATA_FILE)
    n_before, n_fraud_before = len(df), int(df.Class.sum())
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"raw            : {n_before:,} rows, {n_fraud_before} frauds")
        print(f"duplicates     : {n_dup:,} removed")
        print(f"after cleaning : {len(df):,} rows, {int(df.Class.sum())} frauds "
              f"({df.Class.mean()*100:.4f}%)")
    return df


def make_split(df, verbose=True):
    """
    One stratified train/test split, used identically by every notebook.

    Stratification keeps the fraud proportion the same in both parts, which
    matters here because only about 0.17% of transactions are fraudulent
    (Unit I : training, testing and validation of models).
    """
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    if verbose:
        print(f"train : {X_train.shape[0]:,} rows, {int(y_train.sum())} frauds")
        print(f"test  : {X_test.shape[0]:,} rows, {int(y_test.sum())} frauds")
    return X_train, X_test, y_train, y_test


def subsample_train(X, y, n, seed=SEED):
    """
    Reduce the training set for algorithms that cannot handle 199,000 rows
    (k-NN and the RBF-SVM). Every fraud case is kept; only the legitimate
    class is sampled down. The TEST set is never touched.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    fraud = np.flatnonzero(y == 1)
    legit = np.flatnonzero(y == 0)
    take = max(0, min(len(legit), n - len(fraud)))
    idx = np.concatenate([fraud, rng.choice(legit, size=take, replace=False)])
    rng.shuffle(idx)
    return (X.iloc[idx] if hasattr(X, "iloc") else X[idx]), y[idx]


def cv():
    """Stratified k-fold cross validation object (Unit I)."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba=None, unit="", show=True):
    """Compute the standard metrics for one model and optionally print them."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    res = {
        "model": name, "syllabus_unit": unit,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_proba is not None:
        res["pr_auc"] = float(average_precision_score(y_true, y_proba))
        res["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    if show:
        print(f"\n===== {name} =====")
        if unit:
            print(f"syllabus  : {unit}")
        print(f"accuracy  : {res['accuracy']:.5f}")
        print(f"precision : {res['precision']:.4f}")
        print(f"recall    : {res['recall']:.4f}")
        print(f"f1-score  : {res['f1']:.4f}")
        if y_proba is not None:
            print(f"PR-AUC    : {res['pr_auc']:.4f}")
            print(f"ROC-AUC   : {res['roc_auc']:.4f}")
        print(f"confusion : TN={tn:,}  FP={fp}  FN={fn}  TP={tp}")
        print("\n" + classification_report(
            y_true, y_pred, target_names=["Legitimate", "Fraud"],
            digits=4, zero_division=0))
    return res


def save_result(res, y_proba=None):
    """Persist one model's metrics (and scores) so notebook 10 can compare."""
    slug = res["model"].lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    with open(TAB_DIR / f"{slug}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    if y_proba is not None:
        np.save(TAB_DIR / f"proba_{slug}.npy", np.asarray(y_proba, dtype=float))
    print(f"saved result -> Results/tables/{slug}.json")


def load_all_results():
    rows = []
    for f in sorted(TAB_DIR.glob("*.json")):
        if f.name.startswith("dataset"):
            continue
        rows.append(json.loads(f.read_text()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standard plots
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title, fname=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(np.log1p(cm), cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                    color=INK if cm[i, j] < cm.max() * 0.6 else SURFACE)
    ax.set_xticks([0, 1], ["Predicted\nLegitimate", "Predicted\nFraud"])
    ax.set_yticks([0, 1], ["Actual\nLegitimate", "Actual\nFraud"])
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    if fname:
        save_fig(fig, fname)
    return fig


def plot_pr_curve(y_true, y_proba, title, fname=None):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(rec, prec, color=BLUE, label=f"AP = {ap:.4f}")
    base = float(np.mean(y_true))
    ax.axhline(base, color=MUTED, linestyle=":", linewidth=1.4,
               label=f"no-skill = {base:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right"); despine(ax)
    fig.tight_layout()
    if fname:
        save_fig(fig, fname)
    return fig


def plot_roc_curve(y_true, y_proba, title, fname=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(fpr, tpr, color=BLUE, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.4,
            label="random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); despine(ax)
    fig.tight_layout()
    if fname:
        save_fig(fig, fname)
    return fig


HEADER = f"""
Credit Card Fraud Detection using Machine Learning
Lakshay | AP24110010677 | B.Tech CSE | SRM University - AP
Random seed = {SEED} (last two digits of roll number AP24110010677)
"""
