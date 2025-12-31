# utils/metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

ISIC_CLASSES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]

def compute_classwise_metrics(all_targets, all_preds, average='binary'):
    """
    all_targets, all_preds: lists or 1D arrays of ints (0..C-1)
    Returns dict with per-class sensitivity (recall), specificity, precision, f1, accuracy, confusion matrix
    """
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ISIC_CLASSES))))
    # cm[i,i] is TP for class i
    TP = np.diag(cm).astype(float)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN + 1e-12)  # recall
    specificity = TN / (TN + FP + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-12)

    overall_acc = (TP.sum()) / cm.sum()

    per_class = {}
    for i, name in enumerate(ISIC_CLASSES):
        per_class[name] = {
            "sensitivity": float(sensitivity[i]),
            "specificity": float(specificity[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "support": int(cm.sum(axis=1)[i])
        }

    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    overall = {
        "accuracy": float(overall_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist()
    }

    return per_class, overall
