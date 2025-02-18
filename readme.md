# Text Classification Model Training & Evaluation

This repository provides tools for training and evaluating **text classification models** using Logistic Regression 
and **TF-IDF** vectorization. It includes:

- `train.py` → **Train and fine-tune models** using labeled datasets.
- `classify.py` → **Classify text samples or evaluate trained models**.

Training should be fast and generated models light-weight, even on weaker machines. 
I created these scripts with the intention of integrating them as API spam filters for web applications, 
so a low performance overhead is a key concern.

---

## 📂 Features
✅ **Train a model** using a JSON dataset with `text` and `label` fields.
✅ **Fine-tune hyperparameters** for optimal accuracy.
✅ **Evaluate trained models** on test datasets.
✅ **Automatically save the best-performing model**.
✅ **Delete suboptimal models** after fine-tuning.
✅ **Output model metadata for reproducibility**.

---

## 📂 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/KBirenheide/text-classifier-tools
   cd text-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔥 Training a Model (`train.py`)
### **Basic Usage**
To train a model using a labeled dataset and labeled testset:
```bash
python3 train.py dataset.json --name my-classifier --evaluate testset.json
```
### 📜 Input Files
datasets should be in .json form and contain a json array of objects with the alues "text" and "label",
where "text" contains the to-be-classified text, and "label" a binary classification of 1 (true) or 0 (false).
So, if you were to prepare a dataset with a "spam/ham" column, you would map it to "label" as 1 for "spam" 
and 0 for "ham".

### **Optional Arguments**
| Argument | Description |
|----------|-------------|
| `--weight` | Class weight adjustment (default: auto-balanced). |
| `--skip-data-prep` | Skip preprocessing if using a preprocessed dataset. |
| `--common-words N` | Ignore the top N most common words. |
| `--evaluate file.json` | Evaluate the trained model on a test set. |
| `--fine-tune N` | Fine-tune hyperparameters with N steps. |
| `--solver SOLVER` | Choose optimization solver (`liblinear`, `lbfgs`, `saga`, `newton-cg`). |
| `--C VALUE` | Regularization strength (higher = simpler model). |
| `--max-iter N` | Max training iterations (default: 500). |
| `--min-df N` | Minimum document frequency for TF-IDF vectorization. |
| `--max-df VALUE` | Maximum document frequency for TF-IDF. |

### **Fine-Tuning Example**
Fine-tune hyperparameters for better accuracy:
```bash
python3 train.py dataset.json --name my-classifier --evaluate testset.json --fine-tune 5
```

- **Automatically tests different values** for parameters like `C`, `class_weight`, `min_df`, etc.
- **Stores tuning recommendations** in `classifier_my-classifier_meta.json`.

### **Solvers**
| Solver | Type | Best For | Key Characteristics |
|--------|------|----------|---------------------|
| liblinear	Linear | (Coordinate Descent) | Small/Medium Datasets | Fast, handles L1 (Lasso) and L2 (Ridge) regularization |
| lbfgs	| Quasi-Newton | Multiclass, Large Data | Uses second-order derivatives (Hessian), supports L2 regularization |
| saga	| Stochastic Gradient Descent | Large-Scale Sparse Data | Supports L1, L2, and ElasticNet regularization |
| newton-cg	| Newton’s Method | Large Datasets, Multiclass | Uses second-order derivatives for fast convergence, but expensive |

---

## 🎯 Classifying Text (`classify.py`)
### **Classify a Single Text Input**
Use a trained model to classify a single text input:
```bash
python3 classify.py model.joblib --text "This is an example message."
```

### **Classify Multiple Texts from a File**
```bash
python3 classify.py model.joblib --file texts.json
```

- `texts.json` should be a JSON array with `text` fields.

### **Evaluate a Model on a Test Set**
```bash
python3 classify.py model.joblib --evaluate testset.json
```
- Returns the model's **accuracy** and other evaluation metrics.

---

## 📜 Output Files
| File | Description |
|------|-------------|
| `classifier_<name>_<timestamp>.joblib` | Trained model file. |
| `classifier_<name>_meta.json` | Metadata (hyperparameters & evaluation results). |
| `preprocessed_<dataset>.json` | Preprocessed dataset (if used). |

---

## 📢 License & Attribution
- The scripts themselves are licensed under the MIT License, meaninf you can pretty much do 
what you want with them. However, I would appreciate any voluntary attribution when using them.
- The sample models under ./sample-models were created using existing datasets:
    - The **Enron Spam Dataset** was used under **GPL-3.0**, meaning trained models must also be **GPL-3.0 licensed**.
    - The **SMS Spam Dataset** has no specified license, allowing for broader usage.

If you use any of the sample models, please cite the datasets accordingly, the model folders contain the appropriate 
licenses and notices.

🚀 **Happy training & classifying!**

## Example Run

python3 train.py spam_train.json --name spam-ham --evaluate spam_evaluate.json --fine-tune 5 --solver newton-cg --common-words 25
[LOG][INFO] Starting training for classifier: spam-ham
[LOG][INFO] Using dataset: spam_train.json
[LOG][INFO] Dataset loaded successfully.
[LOG][INFO] Final dataset size after cleaning: 20122
[LOG][INFO] Preprocessing dataset...
[LOG][INFO] Most common words excluded: the, to, a, and, of, in, you, for, is, ?, this, i, on, enron, that, it, s, be, with, your, we, !, $, have, from
[LOG][INFO] Preprocessed dataset saved as: preprocessed_spam_train.json
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114159.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 92.91%
[LOG][INFO] Baseline Accuracy: 92.91%
[LOG][INFO] Fine-tuning C parameter.
[LOG][INFO] C parameter fine-tuning step 1 of 5 with parameter value 2.0.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114238.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.27%
[LOG][INFO] C improved! New best: 2.0 (Accuracy: 93.27%)
[LOG][INFO] C parameter fine-tuning step 2 of 5 with parameter value 2.6931471805599454.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114314.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.37%
[LOG][INFO] C improved! New best: 2.6931471805599454 (Accuracy: 93.37%)
[LOG][INFO] C parameter fine-tuning step 3 of 5 with parameter value 3.7917594692280554.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114352.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.45%
[LOG][INFO] C improved! New best: 3.7917594692280554 (Accuracy: 93.45%)
[LOG][INFO] C parameter fine-tuning step 4 of 5 with parameter value 5.178053830347946.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114429.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.53%
[LOG][INFO] C improved! New best: 5.178053830347946 (Accuracy: 93.53%)
[LOG][INFO] C parameter fine-tuning step 5 of 5 with parameter value 6.787491742782047.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114508.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.56%
[LOG][INFO] C improved! New best: 6.787491742782047 (Accuracy: 93.56%)
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114159.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114314.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114429.joblib
[LOG][INFO] Fine-tuning weight parameter.
[LOG][INFO] weight parameter fine-tuning step 1 of 5 with parameter value 1.1.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114546.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.59%
[LOG][INFO] weight improved! New best: 1.1 (Accuracy: 93.59%)
[LOG][INFO] weight parameter fine-tuning step 2 of 5 with parameter value 1.1693147180559946.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114625.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.62%
[LOG][INFO] weight improved! New best: 1.1693147180559946 (Accuracy: 93.62%)
[LOG][INFO] weight parameter fine-tuning step 3 of 5 with parameter value 1.2791759469228057.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114703.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.58%
[LOG][INFO] No improvement, setting step skipping. Optimum tuning for weight is likely between 1.1693147180559946 and 1.2791759469228057
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114238.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114508.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114703.joblib
[LOG][INFO] Fine-tuning min_df parameter.
[LOG][INFO] min_df parameter fine-tuning step 1 of 5 with parameter value 6.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114741.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.46%
[LOG][INFO] No improvement, switching direction. Testing min_df=2
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114821.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] min_df decreased and improved! New best: 2 (Accuracy: 93.84%)
[LOG][INFO] min_df parameter fine-tuning step 2 of 5 with parameter value 1.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-114908.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.58%
[LOG][INFO] No improvement, setting step skipping. Optimum tuning for min_df is likely between 2 and 1
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114352.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114625.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114908.joblib
[LOG][INFO] Fine-tuning max_df parameter.
[LOG][INFO] max_df parameter fine-tuning step 1 of 5 with parameter value 0.9.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-115002.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] No improvement, switching direction. Testing max_df=0.7999999999999999
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-115045.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] max_df parameter fine-tuning step 2 of 5 with parameter value 0.7999999999999999.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-115127.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] No improvement, setting step skipping. Optimum tuning for max_df is likely between 0.85 and 0.7999999999999999
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114546.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-115002.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-115127.joblib
[LOG][INFO] Fine-tuning max_iter parameter.
[LOG][INFO] max_iter parameter fine-tuning step 1 of 5 with parameter value 750.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-115209.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] No improvement, switching direction. Testing max_iter=250
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-115251.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] max_iter parameter fine-tuning step 2 of 5 with parameter value 250.
[LOG][INFO] Saving model to: classifier_spam-ham_20250218-115332.joblib
[LOG][INFO] Starting model evaluation
[LOG][INFO] Model accuracy: 93.84%
[LOG][INFO] No improvement, setting step skipping. Optimum tuning for max_iter is likely between 500 and 250
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Skipping step: fine-tuning this parameter further yields no improvement in either direction.
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-114741.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-115045.joblib
[LOG][INFO] Deleted suboptimal model: classifier_spam-ham_20250218-115251.joblib
[LOG][INFO] Best model retained: classifier_spam-ham_20250218-114821.joblib with accuracy 93.84%.
[LOG][INFO] Model metadata saved as: classifier_spam-ham_20250218-114821.joblib_meta.json