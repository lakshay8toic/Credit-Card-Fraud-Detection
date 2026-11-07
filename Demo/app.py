"""
app.py  -  live demonstration of the fraud detection project.

Run with:
    streamlit run Demo/app.py

The app loads the models that Demo/build_demo.py trained, shows one real
transaction at a time, and asks the chosen model whether it is fraud.
Nothing is trained here - training happened once, in build_demo.py.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
DATA_DIR = HERE / "data"

NAVY, RED, GREEN, ICE, GREY = "#1E2761", "#E34948", "#1BAF7A", "#CADCFC", "#5B6478"

st.set_page_config(page_title="Fraud Detection Demo",
                   page_icon="!", layout="wide")


# ----------------------------------------------------------------------
# Loading (cached so it happens once, not on every click)
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load each saved model. If one fails, carry on with the others.

    XGBoost on macOS needs Apple's OpenMP library, which is a separate
    install (`brew install libomp`). Without it only that one model fails,
    so the demo should still run on the other two rather than crashing.
    """
    out, failed = {}, {}
    for name in ("Decision Tree", "Random Forest", "XGBoost"):
        p = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        if not p.exists():
            continue
        try:
            out[name] = joblib.load(p)
        except Exception as exc:                      # noqa: BLE001
            failed[name] = str(exc).strip().split("\n")[0][:300]
    return out, failed


@st.cache_data
def load_sample():
    df = pd.read_csv(DATA_DIR / "demo_transactions.csv.gz")
    summary = json.loads((DATA_DIR / "summary.json").read_text())
    return df, summary


if not (DATA_DIR / "summary.json").exists():
    st.error("Demo files are missing. Run this first:\n\n"
             "```\npython Demo/build_demo.py\n```")
    st.stop()

models, failed = load_models()
sample, summary = load_sample()
FEATURES = summary["feature_order"]

if not models:
    st.error("No models could be loaded. Run `python Demo/build_demo.py` first.")
    if failed:
        for name, err in failed.items():
            st.caption(f"{name}: {err}")
    st.stop()


# ----------------------------------------------------------------------
# Sidebar - the controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Credit Card Fraud Detection")
    st.caption("B.Tech CSE · SRM University – AP")

    model_name = st.radio("Model", list(models.keys()), index=len(models) - 1,
                          help="All three were trained on the same data, "
                               "with the same split and the same random seed.")
    model = models[model_name]

    st.markdown("---")
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Flag as fraud when probability is at least", 0.01, 0.99, 0.50, 0.01,
        help="The usual default is 0.50. Lowering it catches more fraud but "
             "raises more false alarms.")
    if abs(threshold - 0.50) < 0.005:
        st.caption("0.50 — the textbook default.")
    elif threshold < 0.50:
        st.caption("Below 0.50 — we are saying a missed fraud costs us more "
                   "than a false alarm, so we accept more alerts.")
    else:
        st.caption("Above 0.50 — we only want alerts we are very sure about.")

    st.markdown("---")
    m = summary["models"][model_name]
    st.markdown("**This model on the full test set**")
    st.caption(f"(at the default 0.50 threshold)")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['recall']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c1.metric("F1-score", f"{m['f1']:.3f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    st.caption(f"Caught {m['tp']} of {m['tp'] + m['fn']} frauds · "
               f"{m['fp']} false alarms")

    st.markdown("---")
    st.caption(f"Random seed **{summary['seed']}** · "
               f"{summary['rows_total']:,} transactions · "
               f"{summary['frauds_total']} frauds")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def predict_one(row):
    x = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
    return float(model.predict_proba(x)[0, 1])


def pick(kind):
    if kind == "fraud":
        pool = sample.index[sample.Class == 1]
    elif kind == "legit":
        pool = sample.index[sample.Class == 0]
    else:
        pool = sample.index
    st.session_state.idx = int(np.random.choice(pool))


def explain_note(name):
    if name == "Decision Tree":
        return ("A decision tree is a chain of yes/no questions. "
                "Below is the exact path this transaction took.")
    return ("This model is hundreds of trees voting, so there is no single "
            "rule. Instead we show the features it relies on most overall, "
            "and what this transaction's values were.")


def explain(model, name, row):
    """Return a small table explaining the prediction."""
    if name == "Decision Tree":
        # walk the tree and record each question it asked
        t = model.tree_
        x = row[FEATURES].to_numpy(dtype=float).reshape(1, -1)
        path = model.decision_path(x).indices
        steps = []
        for node in path:
            f = t.feature[node]
            if f == -2:                      # leaf node, no question
                continue
            fname = FEATURES[f]
            thr = t.threshold[node]
            val = float(row[fname])
            went = "yes" if val <= thr else "no"
            steps.append({"question": f"Is {fname} <= {thr:.2f}?",
                          "this transaction": f"{fname} = {val:.2f}",
                          "answer": went})
        return pd.DataFrame(steps)

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame({"note": ["no importances available"]})
    order = np.argsort(imp)[::-1][:8]
    return pd.DataFrame({
        "feature": [FEATURES[i] for i in order],
        "value here": [f"{float(row[FEATURES[i]]):.2f}" for i in order],
        "how much the model uses it": [f"{imp[i]*100:.1f}%" for i in order],
    })


if "idx" not in st.session_state:
    st.session_state.idx = int(np.random.choice(sample.index[sample.Class == 1]))


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    f"<div style='background:{NAVY};padding:18px 24px;border-radius:10px;'>"
    f"<span style='color:white;font-size:26px;font-weight:700;'>"
    f"Credit Card Fraud Detection</span><br>"
    f"<span style='color:{ICE};font-size:15px;'>"
    f"Live demonstration · model currently in use: <b>{model_name}</b></span></div>",
    unsafe_allow_html=True)
st.write("")

if failed:
    names = ", ".join(failed)
    st.warning(
        f"**{names} could not be loaded, so it is not selectable below.** "
        "The other models work normally and the results table in the third "
        "tab still shows every model's scores.\n\n"
        "On macOS this is almost always XGBoost needing Apple's OpenMP "
        "library, which installs separately: `brew install libomp`, then "
        "restart the app.")
    with st.expander("Error detail"):
        for name, err in failed.items():
            st.code(f"{name}: {err}", language="text")

tab1, tab2, tab3 = st.tabs(["Check a transaction",
                            "Run on many transactions",
                            "Why we do not report accuracy"])

# ======================================================================
# TAB 1 - one transaction at a time
# ======================================================================
with tab1:
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    b1.button("Show a real fraud", width="stretch",
              on_click=pick, args=("fraud",))
    b2.button("Show a legitimate one", width="stretch",
              on_click=pick, args=("legit",))
    b3.button("Random transaction", width="stretch",
              on_click=pick, args=("any",))
    b4.number_input("or jump to row", 0, len(sample) - 1,
                    key="idx", step=1)

    row = sample.loc[st.session_state.idx]
    proba = predict_one(row)
    flagged = proba >= threshold
    actual = int(row["Class"])

    left, right = st.columns([3, 2])

    with left:
        colour = RED if flagged else GREEN
        verdict = "FRAUD" if flagged else "LEGITIMATE"
        st.markdown(
            f"<div style='border:3px solid {colour};border-radius:10px;"
            f"padding:22px 26px;'>"
            f"<div style='color:{GREY};font-size:14px;'>Transaction "
            f"#{st.session_state.idx} &nbsp;·&nbsp; Amount "
            f"<b>${row['Amount']:,.2f}</b></div>"
            f"<div style='color:{colour};font-size:46px;font-weight:800;"
            f"line-height:1.25;'>{verdict}</div>"
            f"<div style='color:{GREY};font-size:16px;'>model probability of "
            f"fraud: <b style='color:{colour};'>{proba*100:.2f}%</b> "
            f"&nbsp;(threshold {threshold:.2f})</div></div>",
            unsafe_allow_html=True)

        st.write("")
        # was the model right?
        if flagged and actual == 1:
            st.success("Correct — this really was a fraudulent transaction. "
                       "(true positive)")
        elif not flagged and actual == 0:
            st.success("Correct — this really was a legitimate transaction. "
                       "(true negative)")
        elif flagged and actual == 0:
            st.warning("False alarm — this transaction was actually legitimate. "
                       "(false positive: costs one analyst review)")
        else:
            st.error("Missed — this was a real fraud and we let it through. "
                     "(false negative: costs the bank the full amount)")

    with right:
        st.markdown("**Why the model decided this**")
        st.caption(explain_note(model_name))
        st.dataframe(explain(model, model_name, row), hide_index=True,
                     width="stretch")

    with st.expander("See the raw transaction values"):
        st.dataframe(
            pd.DataFrame({"feature": FEATURES,
                          "value": [row[f] for f in FEATURES]}),
            hide_index=True, width="stretch", height=300)

# ======================================================================
# TAB 2 - run the model over many transactions at once
# ======================================================================
with tab2:
    st.markdown(
        f"Run **{model_name}** over all **{len(sample):,}** transactions in the "
        f"demo sample. The sample contains every fraud from the test set "
        f"(**{int(sample.Class.sum())}** of them) plus "
        f"**{int((sample.Class == 0).sum()):,}** legitimate transactions.")

    if st.button("Run the model", type="primary"):
        with st.spinner("Scoring transactions ..."):
            probs = model.predict_proba(sample[FEATURES])[:, 1]
        st.session_state.probs = probs

    if "probs" in st.session_state:
        probs = st.session_state.probs
        pred = (probs >= threshold).astype(int)
        truth = sample["Class"].to_numpy()

        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        tn = int(((pred == 0) & (truth == 0)).sum())

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Frauds caught", f"{tp} / {tp + fn}")
        k2.metric("Frauds missed", fn)
        k3.metric("False alarms", fp)
        k4.metric("F1-score", f"{f1:.3f}")

        st.write("")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Confusion matrix**")
            cm = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actually legitimate", "Actually fraud"],
                columns=["Predicted legitimate", "Predicted fraud"])
            st.dataframe(cm, width="stretch")
            st.caption(f"At threshold {threshold:.2f}. Move the slider in the "
                       f"sidebar and the numbers change — that is the "
                       f"trade-off between catching fraud and raising alarms.")
        with c2:
            st.markdown("**What the analyst team would see**")
            st.markdown(
                f"- Alerts raised: **{tp + fp:,}**\n"
                f"- Of those, real fraud: **{tp}** "
                f"({precision*100:.1f}% of alerts were worth reviewing)\n"
                f"- Frauds that slipped through: **{fn}**\n"
                f"- Recall: **{recall*100:.1f}%** of all fraud was caught")

        st.markdown("**The transactions the model was most suspicious about**")
        top = sample.copy()
        top["fraud probability"] = probs
        top["actually fraud?"] = np.where(truth == 1, "yes", "no")
        top = (top[["Amount", "fraud probability", "actually fraud?"]]
               .sort_values("fraud probability", ascending=False).head(15))
        top["fraud probability"] = top["fraud probability"].map("{:.1%}".format)
        top["Amount"] = top["Amount"].map("${:,.2f}".format)
        st.dataframe(top, width="stretch", height=330)


# ======================================================================
# TAB 3 - the accuracy trap
# ======================================================================
with tab3:
    n = len(sample)
    n_fraud = int(sample.Class.sum())
    lazy_acc = (n - n_fraud) / n

    st.markdown("### The lazy model")
    st.markdown(
        "Here is a 'model' that ignores the data completely and answers "
        "**legitimate** for every single transaction:")
    st.code("def predict(transaction):\n    return 'legitimate'", language="python")

    c1, c2, c3 = st.columns(3)
    c1.metric("Its accuracy on the real test set",
              f"{(1 - summary['test_frauds']/summary['test_rows'])*100:.2f}%")
    c2.metric("Frauds it catches", "0")
    c3.metric("Its recall", "0.00")

    st.error("It scores over 99.8% accuracy and is completely useless. "
             "This is why accuracy does not appear anywhere in our results.")

    st.markdown("### What we report instead")
    rows = []
    for name, mm in summary["models"].items():
        rows.append({
            "Model": name,
            "Recall": f"{mm['recall']:.3f}",
            "Precision": f"{mm['precision']:.3f}",
            "F1-score": f"{mm['f1']:.3f}",
            "PR-AUC": f"{mm['pr_auc']:.3f}",
            "Frauds caught": f"{mm['tp']} / {mm['tp'] + mm['fn']}",
            "False alarms": mm["fp"],
        })
    # a model that never predicts fraud has PR-AUC equal to the fraud rate
    prevalence = summary["test_frauds"] / summary["test_rows"]
    rows.append({"Model": "Lazy model (always legitimate)", "Recall": "0.000",
                 "Precision": "0.000", "F1-score": "0.000",
                 "PR-AUC": f"{prevalence:.4f}",
                 "Frauds caught": f"0 / {summary['test_frauds']}",
                 "False alarms": 0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Measured on the full held-out test set of "
               f"{summary['test_rows']:,} transactions at the 0.50 threshold. "
               f"Random seed {summary['seed']}.")
