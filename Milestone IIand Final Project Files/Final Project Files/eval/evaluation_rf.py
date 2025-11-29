# evaluation_rf.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

def evaluate_rf_model(y_true, y_pred, y_proba=None,
                      class_names=None,
                      display_confusion_matrix=True,
                      display_roc=True,
                      display_pr=True,
                      figsize=(6,5),
                      title_prefix="Random Forest"):
    """
    Evaluate a RandomForest (or any classifier) predictions and optionally plot evaluation visuals.

    Args:
      y_true (array-like): true labels.
      y_pred (array-like): predicted labels.
      y_proba (array-like or None): predicted scores/probabilities for positive class (for ROC/PR curves).
      class_names (list of str or None): names for classes, e.g. ["Non-toxic", "Toxic"].
      display_confusion_matrix (bool): whether to plot confusion matrix.
      display_roc (bool): whether to plot ROC curve (requires y_proba).
      display_pr (bool): whether to plot PR curve (requires y_proba).
      figsize (tuple): default figure size for plots.
      title_prefix (str): prefix for plot titles / labels.

    Returns:
      results (dict): with keys e.g. 'accuracy', 'classification_report', 'confusion_matrix',
                      and if y_proba given: 'fpr', 'tpr', 'roc_auc', 'precision', 'recall', 'pr_auc'.
    """

    results = {}

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    results['accuracy'] = acc

    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, output_dict=True)

    results['classification_report'] = report

    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm

    if display_confusion_matrix:
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else None,
                    yticklabels=class_names if class_names else None)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f"{title_prefix} — Confusion Matrix")
        plt.show()

    # If probabilities/scores provided — compute ROC & PR curves
    if y_proba is not None:
        y_proba = np.array(y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        results['fpr'], results['tpr'], results['roc_auc'] = fpr, tpr, roc_auc

        if display_roc:
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color='blue', lw=2)
            plt.plot([0,1],[0,1], linestyle='--', color='navy', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{title_prefix} — ROC Curve")
            plt.legend(loc='lower right')
            plt.show()

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        results['precision_curve'], results['recall_curve'], results['pr_auc'] = precision, recall, pr_auc

        if display_pr:
            plt.figure(figsize=figsize)
            plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})", color='green', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f"{title_prefix} — Precision-Recall Curve")
            plt.legend(loc='lower left')
            plt.show()

    return results
