import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    model,
    dataloader,
    device,
    class_names=None,
    model_name: str = "model",
    compute_probs: bool = False
):
    """
    Evaluate a trained model on data from dataloader.
    Prints classification report, confusion matrix, ROC AUC, PR curve, etc.
    Returns a dict of metrics and optionally probabilities.
    
    Args:
        model: a torch.nn.Module, already to(device)
        dataloader: torch.utils.data.DataLoader for test/validation data
        device: torch.device
        class_names: list of class names (for classification report / confusion matrix). If None, uses numeric labels.
        model_name: string, name for this model (used in titles)
        compute_probs: bool, whether to compute predicted probabilities.
                       If False, only hard predictions (argmax) used.
                       If True, must assume model returns logits or probabilities.
    Returns:
        results: dict with keys:
            'y_true', 'y_pred', (if compute_probs) 'y_proba'
            'confusion_matrix', 'classification_report_dict', 'roc_auc', 'pr_auc',
            and optionally ROC/PR curve data: 'fpr', 'tpr', 'precision', 'recall'.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_proba = [] if compute_probs else None

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if compute_probs:
                # assuming outputs are logits
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                y_proba.extend(probs)
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}
    results['y_true'] = y_true
    results['y_pred'] = y_pred

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm

    # Classification report
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print(f"\nClassification Report for {model_name}:\n")
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"\nClassification Report for {model_name}:\n")
        print(classification_report(y_true, y_pred))
    results['classification_report'] = report

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # ROC and PR curves / AUC if probability is available
    if compute_probs and y_proba is not None:
        y_proba = np.array(y_proba)
        results['y_proba'] = y_proba

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
        results['roc_auc'] = roc_auc

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        # Precision‑Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)  # or use average_precision_score
        results['precision'] = precision
        results['recall'] = recall
        results['pr_auc'] = pr_auc

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='green', lw=2,
                 label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision‑Recall Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

    return results