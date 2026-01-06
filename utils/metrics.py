# utils/metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

ISIC_CLASSES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]

def compute_classwise_metrics(y_true, y_pred, class_names=ISIC_CLASSES):
    """
    Returns:
      per_class: dict[class_name] -> {precision, recall (sensitivity), specificity, f1, support}
      overall: dict with accuracy, macro_f1, micro_f1
    """
    if len(y_true) == 0:
        # defensive
        per_class = {c: {"precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1": 0.0, "support": 0} for c in class_names}
        overall = {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}
        return per_class, overall

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for idx, name in enumerate(class_names):
        TP = cm[idx, idx]
        FN = cm[idx, :].sum() - TP
        FP = cm[:, idx].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(TP + FN)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),        # sensitivity
            "specificity": float(specificity),
            "f1": float(f1),
            "support": support
        }

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro")),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    return per_class, overall
