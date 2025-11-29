import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    model,
    dataloader,
    device,
    metadata=None,
    class_names=None,
    model_name: str = "model",
    compute_probs: bool = False,
    show_species_breakdown: bool = True
):
    """
    Evaluate a trained model on data from dataloader.
    Prints classification report, confusion matrix, ROC AUC, PR curve, etc.
    Returns a dict of metrics and optionally probabilities.
    
    Args:
        model: a torch.nn.Module, already to(device)
        dataloader: torch.utils.data.DataLoader for test/validation data
        device: torch.device
        metadata: pandas DataFrame with species info (optional, for species breakdown)
        class_names: list of class names (for classification report / confusion matrix). 
                     If None, uses numeric labels.
        model_name: string, name for this model (used in titles)
        compute_probs: bool, whether to compute predicted probabilities.
        show_species_breakdown: bool, whether to show per-species confusion matrices
        
    Returns:
        results: dict with keys:
            'y_true', 'y_pred', (if compute_probs) 'y_proba'
            'confusion_matrix', 'classification_report_dict', 
            'accuracy', 'precision', 'recall', 'f1_toxic', 'f1_non_toxic'
            'false_negative_rate', 'roc_auc', 'pr_auc',
            and optionally ROC/PR curve data: 'fpr', 'tpr', 'precision_curve', 'recall_curve'
            and if species data available: 'species_fn_rates'
    """
    model.eval()
    y_true = []
    y_pred = []
    y_proba = [] if compute_probs else None
    species_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both (image, label) and (image, label, species) tuples
            if len(batch) == 3:
                inputs, labels, species = batch
                species_list.extend(species)
            else:
                inputs, labels = batch
                
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
    results['accuracy'] = report['accuracy']
    
    # Extract precision, recall, F1 for toxic class
    toxic_label = class_names[1] if class_names else '1'
    non_toxic_label = class_names[0] if class_names else '0'
    
    results['precision'] = report[toxic_label]['precision']
    results['recall'] = report[toxic_label]['recall']
    results['f1_toxic'] = report[toxic_label]['f1-score']
    results['f1_non_toxic'] = report[non_toxic_label]['f1-score']

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

    # False negative rate
    TN, FP, FN, TP = cm.ravel()
    if (FN + TP) > 0:
        fn_rate = FN / (FN + TP)
        results['false_negative_rate'] = fn_rate
        print(f"\nFalse Negative Rate (overall): {fn_rate:.4f}")
    else:
        results['false_negative_rate'] = 0.0

    # Species-level breakdown if available
    if species_list and show_species_breakdown:
        df_test = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'species': species_list
        })
        
        # Get unique species
        unique_species = sorted(set(species_list))
        n_species = len(unique_species)
        fn_species = {}
        
        # Create subplot grid
        cols = 5
        rows = (n_species + cols - 1) // cols  # Ceiling division
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if n_species > 1 else [axes]
        fig.suptitle(f'Confusion Matrices by Species - {model_name}', fontsize=20, y=1.01)
        
        for ax, species in zip(axes, unique_species):
            species_df = df_test[df_test['species'] == species]
            cm_species = confusion_matrix(species_df['y_true'], species_df['y_pred'], labels=[0, 1])
            
            # Calculate FN rate for this species
            if cm_species.size == 4:  # 2x2 matrix
                TN_s, FP_s, FN_s, TP_s = cm_species.ravel()
                if (FN_s + TP_s) > 0:
                    fn_species[species] = FN_s / (FN_s + TP_s)
            
            # Plot
            im = ax.imshow(cm_species, cmap='Greens')
            ax.set_title(species, fontsize=14)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
            
            # Annotate with numbers
            for i in range(cm_species.shape[0]):
                for j in range(cm_species.shape[1]):
                    ax.text(
                        j, i, str(cm_species[i, j]),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='black'
                    )
        
        # Hide unused subplots
        for ax in axes[n_species:]:
            ax.axis('off')
            
        fig.supxlabel('Predicted Values', fontsize=18)
        fig.supylabel('True Values', fontsize=18)
        plt.tight_layout()
        plt.show()
        
        # Print species FN rates
        if fn_species:
            fn_species_df = pd.DataFrame.from_dict(fn_species, orient='index')
            fn_species_df.columns = ['False Negative Rate']
            print("\n" + "="*50)
            print("False Negative Rates by Species:")
            print("="*50)
            print(fn_species_df.to_string(justify='left'))
            results['species_fn_rates'] = fn_species

    # ROC and PR curves / AUC if probability is available
    if compute_probs and y_proba is not None:
        y_proba = np.array(y_proba)
        results['y_proba'] = y_proba

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
        results['roc_auc'] = roc_auc

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='lightblue', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        # Precision‑Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        results['precision_curve'] = precision
        results['recall_curve'] = recall
        results['pr_auc'] = pr_auc

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='lightblue', lw=2,
                 label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision‑Recall Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

    # Print summary
    print("\n" + "="*60)
    print(f"SUMMARY FOR {model_name}")
    print("="*60)
    print(f"Accuracy:              {results['accuracy']:.4f}")
    print(f"Precision (Toxic):     {results['precision']:.4f}")
    print(f"Recall (Toxic):        {results['recall']:.4f}")
    print(f"F1-Score (Toxic):      {results['f1_toxic']:.4f}")
    print(f"F1-Score (Non-Toxic):  {results['f1_non_toxic']:.4f}")
    print(f"False Negative Rate:   {results['false_negative_rate']:.4f}")
    if compute_probs:
        print(f"ROC AUC:               {results['roc_auc']:.4f}")
        print(f"PR AUC:                {results['pr_auc']:.4f}")
    print("="*60)

    return results


def compare_models(results_dict, metric='accuracy'):
    """
    Compare multiple models on a specific metric.
    
    Args:
        results_dict: dict of {model_name: results_dict}
        metric: metric to compare ('accuracy', 'f1_toxic', 'roc_auc', etc.)
    """
    comparison = {}
    for model_name, results in results_dict.items():
        if metric in results:
            comparison[model_name] = results[metric]
    
    # Sort by metric value
    sorted_comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON - {metric.upper()}")
    print(f"{'='*60}")
    for model_name, value in sorted_comparison.items():
        print(f"{model_name:<50} {value:.4f}")
    print(f"{'='*60}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    models = list(sorted_comparison.keys())
    values = list(sorted_comparison.values())
    
    plt.barh(models, values, color='lightblue')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xlim([0, 1.0])
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return sorted_comparison