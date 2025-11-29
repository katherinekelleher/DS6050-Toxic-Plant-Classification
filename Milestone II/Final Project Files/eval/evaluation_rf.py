########################## WIP #######################

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    RocCurveDisplay
)

# Binary Accuracy 
#print(f"Binary accuracy: {acc:.3f}")

# Metrics Report
print("\nMetrics Report:")
print(classification_report(y_val, y_pred, target_names=["Non-toxic (0)", "Toxic (1)"]))
cr = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)

acc = {}
acc['Random Forest Untuned'] = cr['accuracy']
prec = {}
prec['Random Forest Untuned'] = cr['Toxic']['precision']
rec = {}
rec['Random Forest Untuned'] = cr['Toxic']['recall']
f1toxic = {}
f1toxic['Random Forest Untuned'] = cr['Toxic']['f1-score']
f1nontoxic = {}
#f1nontoxic['Random Forest Untuned'] = cr['Non-toxic']['f1-score']

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-toxic", "Toxic"], yticklabels=["Non-toxic", "Toxic"])
plt.title("Confusion Matrix - Random Forest Untuned")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# False negative rate (for toxic species)
TN, FP, FN, TP = cm.ravel()
fn = {}
if FN > 0:
    fn['Random Forest Untuned'] = FN / (FN + TP)
    
# ROC Curve & AUC
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
roc = {}
roc['Random Forest Untuned'] = roc_auc

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='lightblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Untuned')
plt.legend(loc="lower right")
plt.show()

# PR Curve
precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
pr_auc = auc(precision, recall)
pr = {}
pr['Random Forest Untuned'] = pr_auc

plt.figure(figsize=(6,5))
plt.plot(precision, recall, color='lightblue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PR Curve - Random Forest Untuned')
plt.legend(loc="lower right")
plt.show()

# Metrics Report
print("\nMetrics Report (Tuned Model):")
print(classification_report(y_val, y_pred_tuned, target_names=class_names))
cr = classification_report(y_val, y_pred_tuned, target_names=class_names, output_dict=True)
acc['Random Forest Tuned'] = cr['accuracy']
prec['Random Forest Tuned'] = cr['Toxic']['precision']
rec['Random Forest Tuned'] = cr['Toxic']['recall']
f1toxic['Random Forest Tuned'] = cr['Toxic']['f1-score']
#f1nontoxic['Random Forest Tuned'] = cr['Non-toxic']['f1-score']

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_tuned)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-toxic", "Toxic"], yticklabels=["Non-toxic", "Toxic"])
plt.title("Confusion Matrix - Random Forest (Tuned Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# False negative rate (for toxic species)
TN, FP, FN, TP = cm.ravel()
if FN > 0:
    fn['Random Forest Tuned'] = FN / (FN + TP)
    print(f'False Negative Rate: {FN / (FN + TP)}')
    
# ROC Curve & AUC
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_tuned)
roc_auc = auc(fpr, tpr)
roc['Random Forest Tuned'] = roc_auc

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='lightblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f}) (Tuned Model)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (Tuned Model)')
plt.legend(loc="lower right")
plt.show()

# PR Curve
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_tuned)
pr_auc = auc(precision, recall)
pr['Random Forest Tuned'] = pr_auc

plt.figure(figsize=(6,5))
plt.plot(precision, recall, color='lightblue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PR Curve - Random Forest Tuned')
plt.legend(loc="lower right")
plt.show()