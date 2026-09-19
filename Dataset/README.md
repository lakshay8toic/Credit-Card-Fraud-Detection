# Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the
Machine Learning Group of the Université Libre de Bruxelles (ULB).

| Property | Value |
|---|---|
| Source | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Transactions | 284,807 |
| Attributes | 30 (`Time`, `V1`–`V28`, `Amount`) + `Class` |
| Fraudulent | 492 (0.172%) |
| Period | Two days, September 2013, European cardholders |

`V1`–`V28` are principal components produced by applying PCA to the original
transaction attributes, released in that form to protect cardholder
confidentiality. Only `Time`, `Amount` and `Class` are untransformed.

After removing 1,081 exact duplicate rows the project works with **283,726
transactions, of which 473 (0.1667%) are fraudulent**.

## Download

The CSV is about 144 MB and is not committed to this repository (GitHub's file
size limit is 100 MB). Download it and place it in this folder as
`creditcard.csv`.

```bash
# Option A - Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud -p Dataset --unzip

# Option B - download from the URL above and unzip into this folder
```

Verify:

```bash
python -c "import pandas as pd; d=pd.read_csv('Dataset/creditcard.csv'); print(d.shape, d.Class.sum())"
# expected: (284807, 31) 492
```
