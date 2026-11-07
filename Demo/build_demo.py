"""
build_demo.py  -  run this ONCE before the demo.

It trains the three models used by the live demo app and saves them to disk,
together with a small sample of the test set that the app shows on screen.

Why this file exists
--------------------
The notebooks train models and then throw them away. A demo needs the trained
model to still be there afterwards, so this script trains them once and
"pickles" them (saves them as files) using joblib.

Everything here uses exactly the same data, the same cleaning and the same
train/test split as the notebooks, by importing Code/utils.py. So the numbers
the demo shows are the same numbers that are in the report.

The three models use exactly the hyperparameters chosen in notebooks 03 and
08, so the figures this script prints are the figures in the report.

Usage:
    python Demo/build_demo.py              # trains anything not already saved
    python Demo/build_demo.py --retrain    # trains everything again
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

import utils as U                                     # noqa: E402  (shared helpers)
from sklearn.tree import DecisionTreeClassifier       # noqa: E402
from sklearn.ensemble import RandomForestClassifier   # noqa: E402
from sklearn.metrics import (                         # noqa: E402
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, average_precision_score,
)
from xgboost import XGBClassifier                     # noqa: E402

OUT_MODELS = Path(__file__).parent / "models"
OUT_DATA = Path(__file__).parent / "data"
OUT_MODELS.mkdir(exist_ok=True)
OUT_DATA.mkdir(exist_ok=True)

SEED = U.SEED          # 77 - last two digits of roll number AP24110010677
SAMPLE_LEGIT = 1500    # how many legitimate transactions to keep for the demo


def main(retrain=False):
    if not U.DATA_FILE.exists():
        sys.exit(
            f"\nERROR: {U.DATA_FILE} not found.\n"
            "Download creditcard.csv from\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and put it in {U.DATA_FILE.parent}\n")

    print("Loading data ...")
    df = U.load_data()
    X_train, X_test, y_train, y_test = U.make_split(df)

    # ----- the three models the demo lets you switch between ---------------
    # Decision Tree  : Unit II - we can print the exact rule it used
    # Random Forest  : Unit V  - bagging, many trees voting
    # XGBoost        : Unit V  - boosting, the best performer in our tests
    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",        # ID3 uses entropy / information gain
            max_depth=6,                # the depth notebook 03 chose by CV
            min_samples_leaf=20,
            random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",            # the decorrelating restriction
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            # the dataset has ~600 legitimate rows per fraud, so we tell
            # XGBoost to weight the rare class up by roughly that factor
            scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=SEED),
    }

    metrics = {}
    for name, model in models.items():
        path = OUT_MODELS / f"{slug(name)}.joblib"
        if path.exists() and not retrain:
            # already trained on an earlier run - just load it back
            print(f"\n{name}: already trained, loading from disk", flush=True)
            model = joblib.load(path)
            took = 0.0
        else:
            t0 = time.time()
            print(f"\nTraining {name} ...", flush=True)
            model.fit(X_train, y_train)
            took = time.time() - t0
            joblib.dump(model, path, compress=3)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        metrics[name] = {
            "accuracy":  round(float(accuracy_score(y_test, pred)), 6),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "pr_auc":    round(float(average_precision_score(y_test, proba)), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "train_seconds": round(took, 1),
        }
        print(f"  recall {metrics[name]['recall']:.3f}  "
              f"precision {metrics[name]['precision']:.3f}  "
              f"F1 {metrics[name]['f1']:.3f}  PR-AUC {metrics[name]['pr_auc']:.3f}"
              + (f"  ({took:.1f}s)" if took else ""))

    # ----- the sample of transactions the app displays ---------------------
    # Every fraud in the test set, plus a random selection of legitimate ones.
    # This keeps the demo file small (a few hundred KB instead of 144 MB)
    # while still letting us show real frauds on screen.
    rng = np.random.default_rng(SEED)
    fraud_idx = y_test[y_test == 1].index
    legit_idx = rng.choice(y_test[y_test == 0].index.to_numpy(),
                           size=min(SAMPLE_LEGIT, int((y_test == 0).sum())),
                           replace=False)
    keep = np.concatenate([fraud_idx.to_numpy(), legit_idx])
    rng.shuffle(keep)

    sample = X_test.loc[keep].copy()
    sample["Class"] = y_test.loc[keep].values
    sample = sample.reset_index(drop=True)
    sample.to_csv(OUT_DATA / "demo_transactions.csv.gz", index=False,
                  compression="gzip")

    summary = {
        "seed": SEED,
        "rows_total": int(len(df)),
        "frauds_total": int(df.Class.sum()),
        "test_rows": int(len(y_test)),
        "test_frauds": int(y_test.sum()),
        "sample_rows": int(len(sample)),
        "sample_frauds": int(sample.Class.sum()),
        "feature_order": list(X_test.columns),
        "models": metrics,
    }
    (OUT_DATA / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 62)
    print("Demo files written:")
    for p in sorted(OUT_MODELS.glob("*.joblib")):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size/1e6:.1f} MB)")
    for p in sorted(OUT_DATA.iterdir()):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size/1e6:.2f} MB)")
    print("\nNow run:   streamlit run Demo/app.py")
    print("=" * 62)


def slug(name):
    return name.lower().replace(" ", "_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true",
                    help="train again even if a saved model already exists")
    main(retrain=ap.parse_args().retrain)
