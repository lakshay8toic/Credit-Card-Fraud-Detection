# Credit Card Fraud Detection using Machine Learning

**Lakshay** &nbsp;·&nbsp; Roll No: AP24110010677 &nbsp;·&nbsp; B.Tech Computer Science and Engineering
**SRM University – AP** &nbsp;·&nbsp; Supervisor: Mr. Ajay Dilip Kumar Marapatla

A comparative study of machine learning algorithms for detecting fraudulent
credit card transactions, built so that **every unit of the Machine Learning
syllabus is applied to a real problem** rather than studied in isolation.

> **Random seed = 77** throughout — the last two digits of roll number
> AP24110010677, as required. Every result in this repository is reproducible.

---

## The problem in one number

The dataset contains 283,726 transactions, of which **473 (0.1667%) are
fraudulent**. A classifier that ignores its input entirely and predicts
"legitimate" every time therefore scores:

| | |
|---|---|
| **Accuracy** | **99.8333%** |
| **Frauds detected** | **0 of 473** |

Accuracy is meaningless on this problem. Every model below is judged on
**recall**, **precision**, **F1-score** and **PR-AUC** instead.

![Class distribution](Results/figures/01_class_distribution.png)

---

## Results

All twelve models were trained and tested on the **same** stratified split
(70/30, seed 77). The test set contains **85,118 transactions, 142 of them
fraudulent**.

| Model | Syllabus | Accuracy | Precision | Recall | F1 | PR-AUC | Caught | Missed | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| **XGBoost** | V | 0.9996 | 0.9508 | **0.8169** | **0.8788** | **0.8774** | 116 | 26 | 6 |
| AdaBoost | V | 0.9995 | 0.9231 | 0.7606 | 0.8340 | 0.8289 | 108 | 34 | 9 |
| Random Forest | V | 0.9995 | **0.9717** | 0.7254 | 0.8306 | 0.8659 | 103 | 39 | **3** |
| Decision Tree | II | 0.9994 | 0.8672 | 0.7817 | 0.8222 | 0.8075 | 111 | 31 | 17 |
| Neural Network (MLP) | IV | 0.9994 | 0.9204 | 0.7324 | 0.8157 | 0.8498 | 104 | 38 | 9 |
| Voting — fixed rule fusion | V | 0.9994 | 0.8833 | 0.7465 | 0.8092 | — | 106 | 36 | 14 |
| k-Nearest Neighbour | II | 0.9994 | 0.8438 | 0.7606 | 0.8000 | 0.7897 | 108 | 34 | 20 |
| Support Vector Machine | III | 0.9993 | 0.8333 | 0.7394 | 0.7836 | 0.7841 | 105 | 37 | 21 |
| Stacking — trained rule fusion | V | 0.9993 | 0.9474 | 0.6338 | 0.7595 | 0.8375 | 90 | 52 | 5 |
| Perceptron | IV | 0.9991 | 0.6761 | 0.8380 | 0.7484 | — | 119 | 23 | 57 |
| Naive Bayes | III | 0.9895 | 0.1128 | 0.7746 | 0.1970 | 0.1828 | 110 | 32 | 865 |
| K-Means (unsupervised) | V | 0.9904 | 0.1045 | 0.6268 | 0.1791 | 0.1572 | 89 | 53 | 763 |

**Best model: XGBoost** — catches 116 of 142 frauds with only 6 false alarms.

![Model comparison](Results/figures/10_model_comparison.png)

Note the accuracy column. Every model sits within **0.05 percentage points** of
the trivial classifier's 99.8333%, while their recall ranges from 0.63 to 0.84.
That is the entire argument for not using accuracy here.

### Cross-validated check

To confirm the ranking is not an artefact of one split, the main models were
re-evaluated with 5-fold stratified cross validation on the training data:

| Model | CV F1 (mean ± std) | Training time |
|---|---|---|
| Random Forest | 0.8319 ± 0.0127 | 59.0 s |
| XGBoost | 0.8293 ± 0.0304 | 9.0 s |
| Decision Tree | 0.8011 ± 0.0409 | 22.3 s |
| Perceptron | 0.5843 ± 0.2941 | 1.5 s |
| Naive Bayes | 0.2381 ± 0.0167 | 0.7 s |

The ranking holds.

---

## Syllabus coverage

Every one of the five units is applied. **35 syllabus topics** appear across the
notebooks.

![Syllabus coverage](Results/figures/10_syllabus_coverage.png)

| Unit | Topic from the unitization plan | Notebook |
|---|---|---|
| **I** | Types of learning; supervised vs unsupervised | 01, 09 |
| I | Training, testing and validation of models | 01 |
| I | Cross validation | 03, 04, 10 |
| I | Overfitting and underfitting | 01, 03 |
| I | Evaluation of the model | 10 |
| **II** | Decision tree representation, ID3, entropy | 03 |
| II | Hypothesis space search, inductive bias | 03 |
| II | Issues in decision tree learning | 03 |
| II | Instance based learning, k-nearest neighbour | 04 |
| II | Curse of dimensionality | 02, 04 |
| II | Univariate feature selection | 02 |
| II | Multivariate feature selection | 02 |
| II | Feature selection techniques | 02 |
| II | Feature reduction: **PCA** | 02 |
| II | Feature reduction: **LDA** | 02 |
| **III** | Probability and classification, Bayesian learning | 05 |
| III | Naive Bayes | 05 |
| III | **Bayes optimal decisions** | 05 |
| III | SVM: introduction, dual formulation | 06 |
| III | Maximum margin with noise, kernel functions | 06 |
| **IV** | ANN representation, biological motivation | 07 |
| IV | McCulloch-Pitts neuron | 07 |
| IV | Perceptron, perceptron learning | 07 |
| IV | **Logic gates using perceptron** | 07 |
| IV | Problem with perceptron (XOR) | 07 |
| IV | Gradient descent | 07 |
| IV | Multilayer networks, backpropagation | 07 |
| IV | Computational learning theory, VC dimension | 07 |
| **V** | Ensembles: bagging | 08 |
| V | Random Forest | 08 |
| V | Ensembles: boosting | 08 |
| V | **Fixed rule fusion techniques** | 08 |
| V | **Trained rule fusion techniques** | 08 |
| V | Clustering: K-means | 09 |
| V | Hierarchical clustering | 09 |

---

## Notebooks

| # | Notebook | What it contains |
|---|---|---|
| 01 | **Data Exploration and Preprocessing** | Class imbalance, why accuracy fails, duplicate removal, the stratified split, an overfitting/underfitting demonstration |
| 02 | **Feature Selection and Feature Reduction** | Curse of dimensionality measured on the data, filter/wrapper/embedded selection, PCA variance analysis, LDA projection |
| 03 | **Decision Tree Learning (ID3)** | Entropy and information gain computed **by hand**, hypothesis space search, inductive bias, pruning by cross validation |
| 04 | **K-Nearest Neighbour** | Euclidean distance worked step by step and checked against scikit-learn, choosing k, dimensionality effect measured |
| 05 | **Naive Bayes and Bayes Optimal Decisions** | Bayes theorem, the naive assumption tested on the data, the Bayes optimal threshold **derived**, and why calibration is needed to apply it |
| 06 | **Support Vector Machine** | Primal and dual formulations, soft margin, kernel comparison, margin and support vectors visualised |
| 07 | **Artificial Neural Networks** | McCulloch-Pitts neuron, perceptron learning AND/OR/NAND/NOR, **failing on XOR**, gradient descent, backpropagation, VC dimension |
| 08 | **Ensemble Learning** | Bagging, Random Forest, AdaBoost, XGBoost, fixed rule fusion (majority/mean/product/max/min), trained rule fusion (stacking) |
| 09 | **Clustering** | K-means with the elbow method, clusters compared against hidden labels, anomaly scoring, hierarchical clustering and dendrogram |
| 10 | **Model Comparison and Results** | All results collected, cross-validated comparison, syllabus coverage, conclusion |

---

## Some findings worth pointing out

**The perceptron fails on XOR, exactly as the theory says.** It learns AND, OR,
NAND and NOR in a handful of epochs and then never converges on XOR, because no
single hyperplane can separate the classes. A hidden layer solves it.

![Perceptron logic gates](Results/figures/07_perceptron_logic_gates.png)

**The curse of dimensionality is measurable on this dataset.** k-NN performance
*peaks* at a modest number of attributes and then declines as more are added,
even though the extra attributes carry genuine information.

![Dimensionality effect](Results/figures/04_dimensionality_effect.png)

**LDA separates the classes far better than PCA, with one component instead of
two** — which is what the two objectives predict, since PCA maximises variance
and never looks at the label.

![LDA projection](Results/figures/02_lda_projection.png)

**The Bayes optimal threshold is not 0.5.** Deriving it from the loss matrix
gives τ\* = C_FP/(C_FP + C_FN) = 1/51 ≈ 0.0196. Applying it required an extra
step the derivation does not mention: naive Bayes posteriors are not calibrated,
and isotonic calibration was needed before the derived threshold matched the
empirical optimum.

**K-means recovers fraud structure without ever seeing a label.** Clustering the
transactions blind produced clusters with strongly unequal fraud rates.

---

## Reproducing

```bash
git clone https://github.com/lakshay8toic/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt

# place creditcard.csv in Dataset/  (see Dataset/README.md)
jupyter notebook Code/
```

Run the notebooks **in order, 01 through 10**. Notebook 10 reads the result
files the earlier notebooks write into `Results/tables/`.

Every stochastic component is seeded with `RANDOM_STATE = 77` in
`Code/utils.py`, so re-running reproduces every number in this README exactly.

## Repository layout

```
Credit-Card-Fraud-Detection/
├── Code/                  10 notebooks + utils.py (shared data split and metrics)
├── Dataset/               download instructions (the CSV is not committed)
├── Report/                project report (PDF)
├── Presentation/          viva presentation
├── Project Proposal/      project proposal
├── Results/
│   ├── figures/           61 figures, all generated by the notebooks
│   └── tables/            metrics for every model, in CSV and JSON
├── README.md
├── requirements.txt
└── LICENSE
```

## Dataset

Credit Card Fraud Detection, Machine Learning Group — Université Libre de
Bruxelles. Available at
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.

## Licence

MIT — see [LICENSE](LICENSE).
