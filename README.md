# Practical 5 – Ensemble Learning and Performance Evaluation

**TY B.Tech. – Artificial Intelligence and Machine Learning**  
**Course Code:** MDM-AIML-Lab (Unit: Ensemble Learning & Model Generalization)

| Field | Details |
|---|---|
| **Student Name** | Pankaj Shinde |
| **PRN** | 202401100005 |
| **Branch** | Software Engineering |
| **Division / Batch** | A / A1 |
| **Roll No.** | 07 |
| **Faculty In-Charge** | Dr. Khushal Khairnar |
| **Academic Term** | Semester V / A.Y. 2026–2027 |
| **Date of Submission** | 26-10-2026 |

---

## 1. Problem Statement

1. **Implement Stacking Ensemble** using a meta-classifier (Logistic Regression) to combine cross-validated predictions from heterogeneous base classifiers.
2. **Implement an AdaBoost Ensemble** using decision stumps (`max_depth=1`) as weak learners to sequentially minimize exponential loss.
3. **Implement a Voting Ensemble** that combines the predictions of base classifiers through **Majority Voting (Hard Voting)** and **Weighted Probability Averaging (Soft Voting)**.
4. **Compare the performance** of the ensemble models against individual base machine learning models using comprehensive classification metrics (**Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cross-Entropy Log Loss, Zero-One Loss, and Latency**).
5. **Summarize findings**, provide deep insights into when to use each ensemble paradigm (Bagging vs. Boosting vs. Stacking vs. Voting), and suggest potential areas for further research or algorithmic improvement.

---

## 2. Algorithms & Architectural Principles

| Ensemble / Model | Paradigm | Mathematical Principle / Combining Rule | Base / Meta Estimators |
|---|---|---|---|
| **Stacking Ensemble** | Heterogeneous Stacking / Meta-Learning | Trains diverse base learners on original features; aggregates out-of-fold predicted probability matrices as a level-1 feature space; trains a meta-learner to find optimal convex weights. | Base: Decision Tree, SVM, KNN, Gaussian NB, Logistic Reg.<br>**Meta:** Logistic Regression ($K=5$ CV) |
| **AdaBoost Ensemble** | Sequential Boosting | Sequentially trains weak hypotheses $h_m(x)$; reweights training samples giving higher weights $w_i \leftarrow w_i \exp(\alpha_m \mathbb{I}(y_i \neq h_m(x_i)))$ to misclassified instances; final prediction is sign-weighted vote. | Base: Decision Stump (`max_depth=1`), $M=50$ estimators, $\eta=1.0$ |
| **Voting (Hard) Ensemble** | Consensus / Majority Vote | Computes unweighted majority class label among base models: $\hat{y} = \arg\max_c \sum_{m=1}^M \mathbb{I}(\hat{y}_m = c)$. Resolves ties deterministically. | Base: LR, Decision Tree, SVM, KNN, Gaussian NB |
| **Voting (Soft) Ensemble** | Probability Averaging | Averages predicted class conditional posterior probabilities: $P(y=c\|x) = \frac{1}{M}\sum_{m=1}^M P_m(y=c\|x)$ and selects class maximizing average probability. | Base: LR, Decision Tree, SVM, KNN, Gaussian NB |
| **Random Forest (Bagging)** | Parallel Bootstrap Aggregation | Generates $B$ bootstrap replicas of dataset; trains deep unpruned trees with random feature subspace sampling ($\sqrt{d}$); averages predictions to suppress variance. | 100 Trees, `max_depth=5`, $\sqrt{d}$ feature subsampling |
| **Base Classifiers** | Single Estimators | Standalone baseline models: Logistic Regression (convex loss), Decision Tree (Gini splits), SVM (RBF margin), KNN (lazy metric), Gaussian NB (Bayesian likelihood). | Default tuned scikit-learn hyperparameters |

---

## 3. Dataset Description

The models are benchmarked on the clinical **Breast Cancer Wisconsin (Diagnostic) Dataset** (UCI ML Repository):
- **Total Instances:** 569 patient samples
- **Feature Space:** 30 continuous real-valued geometric features computed from digitized FNA images of breast masses (mean, standard error, worst of: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Target Classes:** Binary — Malignant (`0`, 212 samples, 37.3%) vs. Benign (`1`, 357 samples, 62.7%).
- **Data Partitioning:** Stratified 80% Training (455 samples) and 20% Testing (114 samples; 42 malignant, 72 benign).
- **Preprocessing:** Feature standardization using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) fitted strictly on training partition to eliminate data leakage.

---

## 4. Repository Structure

```
MDM_AI-ML_AS05/
├── ensemble_learning.py           # Main executable Python pipeline
├── requirements.txt               # Project dependencies
├── README.md                      # Comprehensive academic documentation
├── sample_io/
│   ├── sample_input.txt           # Dataset and parameter specification
│   ├── sample_output.txt          # Complete execution logs and reports
│   └── results_table.csv          # Structured evaluation metrics in CSV format
└── screenshots/
    ├── 01_ensemble_vs_base_metrics.png       # Grouped bar chart of Accuracy, Precision, Recall, F1
    ├── 02_roc_curves_ensembles_vs_base.png    # Combined multi-model ROC-AUC Curves
    ├── 03_confusion_matrices_ensembles.png    # 2x2 annotated Confusion Matrices for Ensembles
    ├── 04_loss_and_latency_tradeoff.png       # Cross-Entropy Log Loss vs. Inference Latency
    ├── 05_adaboost_learning_stages.png        # AdaBoost stage-wise zero-one error trajectory
    ├── 06_stacking_meta_weights.png           # Learned meta-classifier coefficient weights
    └── 07_decision_boundaries_ensemble_pca.png # 2D PCA decision boundary visualizations
```

---

## 5. Experimental Results & Performance Comparison

The experimental findings across the 114 test instances are summarized in the benchmark table below:

| Model Architecture | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log Loss | Zero-One Loss | Training Time (ms) | Inference Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Support Vector Machine (SVM)** | Base | **0.9825** | **0.9861** | **0.9861** | **0.9861** | 0.9950 | 0.0860 | **0.0175** | 27.56 | 2.65 |
| **Logistic Regression** | Base | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** | **0.0777** | **0.0175** | 16.92 | **0.69** |
| **Stacking Ensemble** | Ensemble | 0.9737 | 0.9726 | **0.9861** | 0.9793 | 0.9927 | 0.0859 | 0.0263 | 9944.84 | 23.59 |
| **Voting (Hard) Ensemble** | Ensemble | 0.9737 | 0.9726 | **0.9861** | 0.9793 | N/A | N/A | 0.0263 | 58.40 | 20.89 |
| **Voting (Soft) Ensemble** | Ensemble | 0.9649 | 0.9595 | **0.9861** | 0.9726 | 0.9907 | 0.0982 | 0.0351 | 63.65 | 18.13 |
| **AdaBoost Ensemble** | Ensemble | 0.9561 | 0.9467 | **0.9861** | 0.9660 | 0.9825 | 0.3982 | 0.0439 | 273.13 | 13.57 |
| **Random Forest (Bagging)** | Ensemble | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9934 | 0.1095 | 0.0439 | 376.85 | 59.48 |
| **K-Nearest Neighbors (KNN)** | Base | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9788 | 0.4073 | 0.0439 | **1.06** | 3.40 |
| **Decision Tree (Single)** | Base | 0.9386 | 0.9577 | 0.9444 | 0.9510 | 0.9342 | 1.6559 | 0.0614 | 9.07 | 0.46 |
| **Gaussian Naïve Bayes** | Base | 0.9298 | 0.9444 | 0.9444 | 0.9444 | 0.9868 | 0.4370 | 0.0702 | 2.98 | 0.68 |

---

## 6. Confusion Matrix & Clinical Diagnostic Analysis

Ground truth counts on test set: **42 Malignant (`0`)**, **72 Benign (`1`)**.

- **Stacking Ensemble:**
  - True Negatives (Malignant): 40 / 42 (95.2%)
  - False Positives: 2 / 42 (4.8%)
  - False Negatives (Missing cancer): **1 / 72 (1.4%)**
  - True Positives (Benign): 71 / 72 (98.6%)
  - *Total Misclassifications:* **3 out of 114 (97.37% Accuracy)**

- **Voting (Hard) Ensemble:**
  - True Negatives: 40 / 42 (95.2%)
  - False Positives: 2 / 42 (4.8%)
  - False Negatives: **1 / 72 (1.4%)**
  - True Positives: 71 / 72 (98.6%)
  - *Total Misclassifications:* **3 out of 114**

- **AdaBoost Ensemble:**
  - True Negatives: 38 / 42 (90.5%)
  - False Positives: 4 / 42 (9.5%)
  - False Negatives: **1 / 72 (1.4%)**
  - True Positives: 71 / 72 (98.6%)
  - *Total Misclassifications:* **5 out of 114**

- **Voting (Soft) Ensemble:**
  - True Negatives: 39 / 42 (92.9%)
  - False Positives: 3 / 42 (7.1%)
  - False Negatives: **1 / 72 (1.4%)**
  - True Positives: 71 / 72 (98.6%)
  - *Total Misclassifications:* **4 out of 114**

> **Clinical Insight:** In oncology diagnosis, False Negatives (predicting Benign when the tumor is Malignant) carry lethal clinical consequences. All ensemble methods achieved a near-perfect Recall of **0.9861** (only 1 false negative), significantly outperforming standalone Decision Trees (4 false negatives) and Naïve Bayes (4 false negatives).

---

## 7. Analysis: When to Use Each Ensemble Method

```
                                 ENSEMBLE PARADIGM MATRIX
┌──────────────────────┬──────────────────────┬────────────────────────┬────────────────────────┐
│ Ensemble Method      │ Primary Objective    │ When to Use (Best Fit) │ Limitations            │
├──────────────────────┼──────────────────────┼────────────────────────┼────────────────────────┤
│ Stacking             │ Maximize accuracy by │ Multi-modal or highly  │ Expensive training;    │
│ (Meta-Learning)      │ learning optimal     │ complex problems with  │ risk of level-1 data   │
│                      │ classifier blending  │ heterogeneous models   │ leakage without CV     │
├──────────────────────┼──────────────────────┼────────────────────────┼────────────────────────┤
│ AdaBoost             │ Reduce Bias in weak  │ High bias base models  │ Highly sensitive to    │
│ (Sequential Boosting)│ learners by focus on │ (stumps, shallow trees)│ noisy outliers;        │
│                      │ hard-to-fit samples  │ Clean, non-noisy data  │ non-parallelizable     │
├──────────────────────┼──────────────────────┼────────────────────────┼────────────────────────┤
│ Voting (Hard / Soft) │ Smooth out variance  │ Quick baseline without │ Sensitive to dominated │
│ (Simple Consensus)   │ across diverse well- │ meta-model training;   │ or poorly calibrated   │
│                      │ calibrated models    │ production inference   │ probability estimators │
├──────────────────────┼──────────────────────┼────────────────────────┼────────────────────────┤
│ Random Forest        │ Reduce Variance in   │ High variance models   │ Slower inference with  │
│ (Parallel Bagging)   │ deep decision trees  │ (deep trees); high-dim │ hundreds of trees;     │
│                      │ via bootstrap decorr.│ tabular feature spaces │ opaque decision rules  │
└──────────────────────┴──────────────────────┴────────────────────────┴────────────────────────┘
```

1. **When to use Stacking:**
   - Use when you have multiple radically different base classifiers (e.g. neural nets, gradient boosted trees, linear models, KNN) that make independent, uncorrelated errors.
   - Stacking enables a higher-level meta-estimator to automatically discover which base model is reliable in specific sub-regions of feature space.

2. **When to use AdaBoost:**
   - Use when base learners are underfitting (weak learners like decision stumps).
   - Boosting aggressively reduces bias by directing successive estimators to focus entirely on instances misclassified by earlier iterations.

3. **When to use Voting (Hard/Soft):**
   - Use **Soft Voting** when individual classifiers output well-calibrated posterior probabilities. It gives higher weight to confident models.
   - Use **Hard Voting** when probability calibration cannot be guaranteed or models only output discrete class decisions.

4. **When to use Bagging / Random Forest:**
   - Use when individual estimators suffer from high variance and overfitting (e.g., deep unpruned decision trees). Bagging reduces variance exponentially through bootstrap aggregation.

---

## 8. Areas for Further Research & Improvements

1. **Multi-Level Hierarchical Stacking:** Exploring Level-2 and Level-3 meta-learners combining predictions with original features (feature pass-through).
2. **Modern Gradient Boosting Paradigms:** Benchmarking advanced second-order gradient boosting frameworks including **XGBoost**, **LightGBM**, and **CatBoost** against classic AdaBoost.
3. **Automated Dynamic Classifier Selection (DCS):** Implementing Dynamic Ensemble Selection (DES-P, KNORA) where base classifiers are selected dynamically at inference time based on local neighborhood competence.
4. **Calibration-Aware Soft Voting:** Applying Isotonic Regression or Temperature Scaling to base classifiers before probability averaging to eliminate overconfident calibration errors.

---

## 9. How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pankaj-Shinde-07/MDM_AI-ML_AS05.git
   cd MDM_AI-ML_AS05
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the pipeline:**
   ```bash
   python ensemble_learning.py
   ```
   All diagnostic plots will be automatically written to `screenshots/` and experimental metrics saved to `sample_io/`.
