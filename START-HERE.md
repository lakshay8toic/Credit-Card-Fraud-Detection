# Start Here

A short guide to this repository: what to look at first, what each notebook
shows, and plain-English answers to the questions this project tends to attract.

---

## What the project does

It decides whether a credit card transaction is fraudulent. The difficulty is
not the decision itself — it is that fraud is only **0.17%** of the data.

> A model that answers "legitimate" for every single transaction scores
> **99.83% accuracy** and catches **zero** fraud.

That one fact drives every choice in this project: it is why accuracy is never
reported, why recall and precision are used instead, and why the decision
threshold is chosen from cost rather than left at 0.5.

---

## See it working

```bash
pip install -r requirements.txt -r Demo/requirements.txt
# put creditcard.csv into Dataset/   (see Dataset/README.md)
python Demo/build_demo.py        # trains and saves the models, ~1 minute
streamlit run Demo/app.py        # opens the demo in your browser
```

Full instructions in [`Demo/README.md`](Demo/README.md).

---

## The six notebooks that matter

Read these in order. Together they are the whole project.

| # | Notebook | What it shows | The one thing it proves |
|---|---|---|---|
| 01 | Data Exploration and Preprocessing | The data, the imbalance, the cleaning | 473 frauds in 283,726 transactions — and why that breaks accuracy |
| 03 | Decision Tree Learning (ID3) | A tree built with entropy and information gain | A model whose every decision can be read as a rule |
| 04 | K-Nearest Neighbour | Instance-based learning, no training phase | Performance falls as irrelevant features are added |
| 05 | Naive Bayes and Bayes Optimal Decisions | Bayes theorem applied to transactions | Where the decision threshold should sit once cost is known |
| 08 | Ensemble Learning | Bagging, Random Forest, AdaBoost, XGBoost | Many weak models beat one strong model |
| 10 | Model Comparison and Results | Every model on one identical test set | The final ranking, and that it holds under cross validation |

### Supporting notebooks

| # | Notebook | Why it is here |
|---|---|---|
| 06 | Support Vector Machine | Unit III — margin-based classification |
| 07 | Artificial Neural Networks | Unit IV — perceptron, the XOR limitation, backpropagation |

### `Code/Extras/`

Two further analyses that support the main work but are not part of the core
argument. They are kept because the report refers to them.

| # | Notebook | Why it is in Extras |
|---|---|---|
| 02 | Feature Selection and Feature Reduction | Supporting analysis — which features matter, and PCA / LDA |
| 09 | Clustering | Unsupervised check — can fraud be found without labels at all? |

---

## Questions this project attracts

Short answers, in plain language.

**Why not report accuracy?**
Because at 0.17% fraud, answering "legitimate" every time already scores
99.83%. Accuracy measures the wrong thing when one class is this rare.

**Why PR-AUC and not ROC-AUC?**
ROC-AUC uses the false positive rate, whose denominator is the enormous
legitimate class. That denominator swamps everything, so ROC-AUC looks good
even for a weak model. PR-AUC uses precision, which reacts to false positives
directly.

**What is data leakage, and where could it happen here?**
Leakage is test information reaching the model during training. Three places
it could happen here: scaling before splitting, oversampling before splitting,
and duplicate rows landing in both halves. We deduplicate first, split second,
and keep scaling and resampling inside the pipeline so they only ever see the
training fold.

**What is the Bayes optimal decision threshold?** *(Unit III, notebook 05)*
If missing a fraud costs far more than a false alarm, you should not wait until
you are 50% sure before acting. With a false negative costing 50 times a false
positive, the break-even point is 1/51, about 0.02. So the model flags anything
above 2% probability instead of 50%.

**Why did you have to calibrate the probabilities first?**
Because that threshold calculation assumes the model outputs real
probabilities. Naive Bayes outputs numbers that are pushed very close to 0 and
1, so its "90%" is not really 90%. Isotonic calibration rescales those outputs
to match observed frequencies, after which the threshold behaves as the theory
predicts.

**What is the difference between PCA and LDA?** *(Extras, notebook 02)*
PCA finds directions with the most spread in the data and ignores the labels.
LDA finds directions that best separate the two classes, so it uses the labels.
PCA is unsupervised, LDA is supervised.

**What is VC dimension?** *(Unit IV, notebook 07)*
A measure of how complex a model is — the largest number of points it can
separate in every possible arrangement. It is used to estimate how much
training data a model of that complexity needs before its test performance can
be trusted.

**What is the difference between bagging and boosting?** *(Unit V, notebook 08)*
Bagging trains many models independently on random subsets and averages them;
it reduces variance. Boosting trains models one after another, each one
focusing on what the previous got wrong; it reduces bias. Random Forest is
bagging, AdaBoost and XGBoost are boosting.

**What is stacking?** *(Unit V, notebook 08)*
Instead of combining model predictions with a fixed rule like majority vote, a
second model is trained to learn the best way to combine them.

**Why seed 77?**
It is the last two digits of roll number AP24110010677, as instructed. It fixes
the train/test split and every random choice, so anyone running this repository
gets exactly the numbers in the report.

---

## Repository map

```
Code/              the notebooks, and utils.py which every notebook imports
Code/Extras/       supporting analyses
Demo/              the live demonstration app
Dataset/           where creditcard.csv goes (not committed, 144 MB)
Report/            the project report
Presentation/      the slide deck
Project Proposal/  the proposal
Results/           every figure and results table the notebooks produced
```

`Code/utils.py` is the reason the comparison is fair: it holds the single data
loader, the single train/test split and the single set of metrics, so every
notebook is measured the same way.
