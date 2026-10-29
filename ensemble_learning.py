"""
Practical Assignment 5: Ensemble Learning and Performance Evaluation
Course: TY B.Tech - Artificial Intelligence and Machine Learning (MDM-AIML-Lab)
Student: Pankaj Shinde (PRN: 202401100005, Roll No: 07, Batch: A1)
Faculty: Dr. Khushal Khairnar

This script implements:
1. Base Classifiers: Logistic Regression, Support Vector Machine (SVM), Decision Tree, KNN, Gaussian Naive Bayes
2. Stacking Ensemble using Logistic Regression meta-classifier with cross-validation
3. AdaBoost Ensemble with decision stumps and stage-wise error analysis
4. Voting Ensemble (both Hard Majority Voting and Soft Probability Voting)
5. Random Forest (Bagging benchmark)
6. Comprehensive performance comparison across classification metrics, loss functions, and execution latencies
7. Generation of 7 publication-grade diagnostic plots and structured sample I/O files
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    zero_one_loss,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Base classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Ensemble methods
from sklearn.ensemble import (
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier
)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#BDC3C7'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
SAMPLE_IO_DIR = os.path.join(BASE_DIR, "sample_io")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_IO_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Loads Breast Cancer Wisconsin dataset and applies stratified split and feature scaling."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names  # 0: malignant, 1: benign

    # 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardization using StandardScaler fitted ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled, y_train, y_test,
        feature_names, target_names, scaler, X, y
    )


def initialize_models():
    """Initializes base classifiers and ensemble architectures."""
    # 1. Base Classifiers
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        "Gaussian Naïve Bayes": GaussianNB()
    }

    # Selected diverse base estimators for ensembles
    estimators_for_ensemble = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)),
        ('nb', GaussianNB())
    ]

    # 2. Ensemble Models
    ensemble_models = {
        # Task 1: Stacking ensemble using Logistic Regression meta-classifier
        "Stacking Ensemble": StackingClassifier(
            estimators=estimators_for_ensemble,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        ),
        # Task 2: AdaBoost ensemble with decision stumps
        "AdaBoost Ensemble": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ),
        # Task 3a: Hard Voting Ensemble (Majority Voting)
        "Voting (Hard) Ensemble": VotingClassifier(
            estimators=estimators_for_ensemble,
            voting='hard'
        ),
        # Task 3b: Soft Voting Ensemble (Weighted Probability Averaging)
        "Voting (Soft) Ensemble": VotingClassifier(
            estimators=estimators_for_ensemble,
            voting='soft'
        ),
        # Benchmark: Random Forest (Bagging)
        "Random Forest (Bagging)": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
    }

    return base_models, ensemble_models, estimators_for_ensemble


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Trains a model and records extensive performance and latency metrics."""
    # Measure Training Time
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_ms = (time.perf_counter() - start_train) * 1000

    # Measure Inference Time
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time_ms = (time.perf_counter() - start_infer) * 1000

    # Metric computations
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    zero_one = zero_one_loss(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Probabilistic metrics (Log Loss, ROC-AUC)
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob[:, 1])
            ce_loss = log_loss(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        except Exception:
            y_prob = None
            auc = np.nan
            ce_loss = np.nan
            fpr, tpr = None, None
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)
        ce_loss = np.nan
        fpr, tpr, _ = roc_curve(y_test, scores)
        y_prob = None
    else:
        # For hard voting classifier where predict_proba is disabled
        auc = np.nan
        ce_loss = np.nan
        fpr, tpr = None, None
        y_prob = None

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "log_loss": ce_loss,
        "zero_one_loss": zero_one,
        "roc_auc": auc,
        "train_time_ms": train_time_ms,
        "infer_time_ms": infer_time_ms,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def generate_visualizations(results_df, results_dict, X_train, y_train, X_test, y_test):
    """Generates 7 publication-grade diagnostic plots."""
    print("\n--- Generating Visualizations ---")

    # 1. Bar Chart: Comparative Performance Metrics (Accuracy, Precision, Recall, F1)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plot_df = results_df.copy()
    plot_df = plot_df.set_index("Model")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    colors = ["#2B5C8F", "#27AE60", "#E67E22", "#8E44AD"]
    plot_df[metrics_to_plot].plot(kind="bar", ax=ax, color=colors, width=0.8, edgecolor='black', linewidth=0.6)
    
    ax.set_title("Performance Metrics Comparison: Base Models vs. Ensemble Architectures", fontsize=13, fontweight='bold', pad=14, color='#154360')
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Classification Algorithm / Ensemble Technique", fontsize=11, fontweight='bold')
    ax.set_ylim(0.85, 1.01)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.xticks(rotation=25, ha='right', fontsize=9.5)
    plt.tight_layout()
    fig1_path = os.path.join(SCREENSHOTS_DIR, "01_ensemble_vs_base_metrics.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # 2. Combined ROC Curves
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    palette = sns.color_palette("tab10", len(results_dict))
    
    for i, (name, res) in enumerate(results_dict.items()):
        if res["fpr"] is not None and res["tpr"] is not None and not np.isnan(res["roc_auc"]):
            ax.plot(
                res["fpr"], res["tpr"],
                label=f"{name} (AUC = {res['roc_auc']:.4f})",
                linewidth=2.0 if "Ensemble" in name or "Forest" in name else 1.2,
                linestyle="-" if "Ensemble" in name or "Forest" in name else "--",
                color=palette[i]
            )

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random Chance (AUC = 0.5000)')
    ax.set_title("Receiver Operating Characteristic (ROC) Curves: Base Models vs. Ensembles", fontsize=13, fontweight='bold', pad=14, color='#154360')
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc='lower right', fontsize=8.5, frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig2_path = os.path.join(SCREENSHOTS_DIR, "02_roc_curves_ensembles_vs_base.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # 3. Annotated Confusion Matrices for the 4 Ensemble Models + Best Base Model
    target_models = ["Stacking Ensemble", "AdaBoost Ensemble", "Voting (Soft) Ensemble", "Random Forest (Bagging)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes = axes.flatten()

    for idx, model_name in enumerate(target_models):
        res = results_dict[model_name]
        cm = res["confusion_matrix"]
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=['Malignant (0)', 'Benign (1)'],
            yticklabels=['Malignant (0)', 'Benign (1)']
        )
        ax.set_title(f"{model_name}\nAcc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}", fontsize=11, fontweight='bold', color='#154360')
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=10, fontweight='bold')

    plt.suptitle("Confusion Matrix Diagnostics for Ensemble Learning Architectures", fontsize=14, fontweight='bold', y=0.99, color='#154360')
    plt.tight_layout()
    fig3_path = os.path.join(SCREENSHOTS_DIR, "03_confusion_matrices_ensembles.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # 4. Trade-off: Log Loss vs Inference Latency
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    scatter_df = results_df.dropna(subset=["Log Loss"]).copy()
    
    for _, row in scatter_df.iterrows():
        is_ens = "Ensemble" in row["Model"] or "Forest" in row["Model"]
        ax.scatter(
            row["Inference Latency (ms)"], row["Log Loss"],
            s=220 if is_ens else 140,
            color="#C0392B" if is_ens else "#2980B9",
            alpha=0.85,
            edgecolors='black',
            linewidths=1.2,
            zorder=3
        )
        ax.annotate(
            row["Model"],
            (row["Inference Latency (ms)"], row["Log Loss"]),
            xytext=(6, 4), textcoords='offset points',
            fontsize=8.5, fontweight='bold' if is_ens else 'normal',
            color='#1B263B'
        )

    ax.set_title("Cross-Entropy Log Loss vs. Inference Latency Trade-Off Analysis", fontsize=13, fontweight='bold', pad=14, color='#154360')
    ax.set_xlabel("Inference Latency (Milliseconds, ms)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cross-Entropy Log Loss (Lower is Better)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig4_path = os.path.join(SCREENSHOTS_DIR, "04_loss_and_latency_tradeoff.png")
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f"Saved: {fig4_path}")

    # 5. AdaBoost Iterative Learning Curve (Stage-wise Error Progression)
    print("Computing AdaBoost stage-wise convergence...")
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    ada.fit(X_train, y_train)

    train_errors = [1.0 - accuracy_score(y_train, y_pred) for y_pred in ada.staged_predict(X_train)]
    test_errors = [1.0 - accuracy_score(y_test, y_pred) for y_pred in ada.staged_predict(X_test)]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    iterations = range(1, len(train_errors) + 1)
    ax.plot(iterations, train_errors, label="Training Zero-One Error", color="#2980B9", lw=2)
    ax.plot(iterations, test_errors, label="Testing Zero-One Error", color="#E74C3C", lw=2, linestyle='--')
    
    min_test_err_idx = int(np.argmin(test_errors))
    ax.scatter(min_test_err_idx + 1, test_errors[min_test_err_idx], color="#27AE60", s=130, zorder=5, label=f"Min Test Error ({test_errors[min_test_err_idx]:.4f} at iter {min_test_err_idx+1})")
    
    ax.set_title("AdaBoost Sequential Boosting Error Trajectory Across 100 Estimators", fontsize=13, fontweight='bold', pad=14, color='#154360')
    ax.set_xlabel("Boosting Iterations (Number of Weak Estimator Stumps)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Zero-One Classification Error", fontsize=11, fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig5_path = os.path.join(SCREENSHOTS_DIR, "05_adaboost_learning_stages.png")
    fig.savefig(fig5_path)
    plt.close(fig)
    print(f"Saved: {fig5_path}")

    # 6. Stacking Architecture: Meta-Classifier Feature Importance / Coefficients
    stacking_model = results_dict["Stacking Ensemble"]["model"]
    meta_clf = stacking_model.final_estimator_
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    estimator_names = [name for name, _ in stacking_model.estimators]
    
    # In binary classification, meta-classifier (LogisticRegression) learns coefficients for each base model output
    coefs = meta_clf.coef_[0]
    
    # Bar plot of learned weights
    bar_colors = ["#27AE60" if c > 0 else "#C0392B" for c in coefs]
    bars = ax.barh(estimator_names, coefs, color=bar_colors, edgecolor='black', linewidth=0.6, height=0.55)
    
    for bar in bars:
        width = bar.get_width()
        offset = 0.05 if width >= 0 else -0.2
        ax.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                va='center', ha='left' if width >= 0 else 'right', fontsize=9.5, fontweight='bold')

    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_title("Stacking Meta-Classifier (Logistic Regression) Learned Base Model Weights", fontsize=13, fontweight='bold', pad=14, color='#154360')
    ax.set_xlabel("Meta-Learner Weight Coefficient (Impact on Final Ensemble Prediction)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Base Estimator", fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig6_path = os.path.join(SCREENSHOTS_DIR, "06_stacking_meta_weights.png")
    fig.savefig(fig6_path)
    plt.close(fig)
    print(f"Saved: {fig6_path}")

    # 7. 2D PCA Decision Boundaries: Base Models vs Ensembles
    print("Projecting to 2D PCA and computing decision boundaries...")
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train 4 representative models on 2D PCA representation
    pca_models = {
        "Decision Tree (Base)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "SVM RBF (Base)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        "AdaBoost (Ensemble)": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=50, random_state=42),
        "Voting Soft (Ensemble)": VotingClassifier(
            estimators=[
                ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
                ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('lr', LogisticRegression(random_state=42))
            ],
            voting='soft'
        )
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    x_min, x_max = X_test_pca[:, 0].min() - 1.0, X_test_pca[:, 0].max() + 1.0
    y_min, y_max = X_test_pca[:, 1].min() - 1.0, X_test_pca[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    for idx, (p_name, p_clf) in enumerate(pca_models.items()):
        p_clf.fit(X_train_pca, y_train)
        Z = p_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax = axes[idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = ax.scatter(
            X_test_pca[:, 0], X_test_pca[:, 1],
            c=y_test, cmap=plt.cm.coolwarm, edgecolors='k', s=35, alpha=0.85
        )
        test_acc = accuracy_score(y_test, p_clf.predict(X_test_pca))
        ax.set_title(f"{p_name}\n2D PCA Test Accuracy: {test_acc:.4f}", fontsize=11, fontweight='bold', color='#154360')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=9.5, fontweight='bold')
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=9.5, fontweight='bold')

    plt.suptitle("Empirical Decision Boundary Comparison in Reduced 2D PCA Space", fontsize=14, fontweight='bold', y=0.99, color='#154360')
    plt.tight_layout()
    fig7_path = os.path.join(SCREENSHOTS_DIR, "07_decision_boundaries_ensemble_pca.png")
    fig.savefig(fig7_path)
    plt.close(fig)
    print(f"Saved: {fig7_path}")


def main():
    print("="*80)
    print(" PRACTICAL ASSIGNMENT 5: ENSEMBLE LEARNING & PERFORMANCE EVALUATION")
    print(" Student: Pankaj Shinde | PRN: 202401100005 | Roll No: 07 | Division: A (A1)")
    print(" Faculty: Dr. Khushal Khairnar | TY B.Tech AIML")
    print("="*80)

    # 1. Load data
    (
        X_train, X_test, y_train, y_test,
        feature_names, target_names, scaler, X_raw, y_raw
    ) = load_and_preprocess_data()

    print(f"\n[+] Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"[+] Total samples: {X_raw.shape[0]}, Features: {X_raw.shape[1]}")
    print(f"[+] Class distribution: Malignant (0) = {np.sum(y_raw==0)}, Benign (1) = {np.sum(y_raw==1)}")
    print(f"[+] Train partition: {X_train.shape[0]} samples (80%), Test partition: {X_test.shape[0]} samples (20%)")

    # 2. Initialize models
    base_models, ensemble_models, _ = initialize_models()
    all_models = {**base_models, **ensemble_models}

    # 3. Train and evaluate all models
    results_list = []
    results_dict = {}

    print("\n--- Training and Evaluating Models ---")
    for name, model in all_models.items():
        print(f"Evaluating: {name}...")
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results_dict[name] = res
        results_list.append({
            "Model": name,
            "Category": "Ensemble" if ("Ensemble" in name or "Forest" in name) else "Base Classifier",
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1"],
            "ROC-AUC": res["roc_auc"],
            "Log Loss": res["log_loss"],
            "Zero-One Loss": res["zero_one_loss"],
            "Training Time (ms)": res["train_time_ms"],
            "Inference Latency (ms)": res["infer_time_ms"]
        })

    results_df = pd.DataFrame(results_list)

    # Format console output table
    print("\n" + "="*115)
    print(f"{'Model':<26} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'ROC-AUC':<8} | {'LogLoss':<8} | {'Train(ms)':<9} | {'Infer(ms)':<9}")
    print("-" * 115)
    for _, row in results_df.iterrows():
        auc_str = f"{row['ROC-AUC']:.4f}" if not np.isnan(row['ROC-AUC']) else "N/A"
        ll_str = f"{row['Log Loss']:.4f}" if not np.isnan(row['Log Loss']) else "N/A"
        print(f"{row['Model']:<26} | {row['Accuracy']:<8.4f} | {row['Precision']:<9.4f} | {row['Recall']:<7.4f} | {row['F1-Score']:<7.4f} | {auc_str:<8} | {ll_str:<8} | {row['Training Time (ms)']:<9.2f} | {row['Inference Latency (ms)']:<9.2f}")
    print("=" * 115)

    # 4. Save results to CSV
    csv_path = os.path.join(SAMPLE_IO_DIR, "results_table.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Results table exported to: {csv_path}")

    # 5. Save sample input & output files
    input_txt_path = os.path.join(SAMPLE_IO_DIR, "sample_input.txt")
    with open(input_txt_path, "w", encoding="utf-8") as f:
        f.write("PRACTICAL 5: ENSEMBLE LEARNING - SAMPLE INPUT SPECIFICATION\n")
        f.write("="*70 + "\n")
        f.write("Dataset: Breast Cancer Wisconsin (Diagnostic)\n")
        f.write(f"Number of Instances: {X_raw.shape[0]}\n")
        f.write(f"Number of Features: {X_raw.shape[1]} (All continuous geometric descriptors)\n")
        f.write("Target Classes: 0: Malignant (212 instances), 1: Benign (357 instances)\n")
        f.write("Train/Test Ratio: 80% Training (455 samples) / 20% Testing (114 samples)\n")
        f.write("Preprocessing: Scikit-learn StandardScaler (Z-score normalization zero mean unit var)\n")
        f.write("\nFeatures List:\n")
        for idx, feat in enumerate(feature_names, 1):
            f.write(f"  {idx:2d}. {feat}\n")
        f.write("\nEnsemble Implementations Evaluated:\n")
        f.write("  1. Stacking Ensemble: Base models [DT, SVM, KNN, NB, LR] -> Meta-Learner [Logistic Regression]\n")
        f.write("  2. AdaBoost Ensemble: Base estimator [DecisionTree(max_depth=1)] -> 50 estimators, LR=1.0\n")
        f.write("  3. Voting Ensemble (Hard): Majority voting rule over 5 base estimators\n")
        f.write("  4. Voting Ensemble (Soft): Weighted probability averaging over 5 base estimators\n")
        f.write("  5. Random Forest (Bagging benchmark): 100 decision trees with feature sub-sampling\n")

    output_txt_path = os.path.join(SAMPLE_IO_DIR, "sample_output.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("PRACTICAL 5: ENSEMBLE LEARNING - SAMPLE EXECUTION OUTPUT LOG\n")
        f.write("="*85 + "\n")
        f.write("Execution Environment: Python 3.10+, Scikit-Learn 1.0+\n")
        f.write("Student: Pankaj Shinde (PRN: 202401100005, Roll No: 07)\n\n")
        f.write("EXPERIMENTAL BENCHMARK TABLE:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*85 + "\n")
        f.write("DETAILED CLASSIFICATION REPORTS:\n\n")
        for name, res in results_dict.items():
            f.write(f"--- Model: {name} ---\n")
            f.write(f"Accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall:    {res['recall']:.4f}\n")
            f.write(f"F1-Score:  {res['f1']:.4f}\n")
            if not np.isnan(res['roc_auc']):
                f.write(f"ROC-AUC:   {res['roc_auc']:.4f}\n")
            if not np.isnan(res['log_loss']):
                f.write(f"Log Loss:  {res['log_loss']:.4f}\n")
            f.write(f"Zero-One:  {res['zero_one_loss']:.4f}\n")
            f.write(f"Train Time: {res['train_time_ms']:.2f} ms | Infer Latency: {res['infer_time_ms']:.2f} ms\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(res['confusion_matrix']))
            f.write("\n\n")

    print(f"[+] Sample input logged to: {input_txt_path}")
    print(f"[+] Sample output logged to: {output_txt_path}")

    # 6. Generate visualizations
    generate_visualizations(results_df, results_dict, X_train, y_train, X_test, y_test)

    print("\n[SUCCESS] Practical 5 execution and artifact generation completed successfully!")


if __name__ == "__main__":
    main()
