import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)


# Label 0 --> Vanilla Benign
# Label 1 --> Vanilla Harmful
# Label 2 --> Adversarial Benign
# Label 3 --> Adversarial Harmful
def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_cls = len(np.unique(y_true))
    labels = np.arange(0, n_cls)

    # Confusion matrix
    acc = accuracy_score(y_true, y_pred)
    auc = group_auc(y_true, y_pred, group=(1, 3))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Malicious macro F1 (only for malicious classes), handle zero divisions gracefully
    mal_labels = [1, 3]
    mal_macro_f1 = f1_score(
        y_true, 
        y_pred, 
        labels=mal_labels, 
        average='macro', 
        zero_division=0
    )

    # Adversarial macro F1 (only for adversarial classes)
    adv_labels = [2, 3]
    adv_macro_f1 = f1_score(
        y_true, 
        y_pred, 
        labels=adv_labels, 
        average='macro', 
        zero_division=0
    )
    
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Class 0 --> vanilla benign")
    print("Class 1 --> vanilla harmful")
    print("Class 2 --> adversarial benign")
    print("Class 3 --> adversarial harmful\n")
    
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels), "\n")
    
    print(f"Harmful Macro F1: {mal_macro_f1:.4f}")
    print(f"Adversarial Macro F1: {adv_macro_f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels, 
        zero_division=0
    ))

    return {
        'accuracy': acc,
        'harmful_macro_f1': mal_macro_f1,
        'adv_macro_f1': adv_macro_f1,
        'auc': auc,
        'conf_mat': cm.tolist()
    }


def group_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: tuple[int, ...] = (1, 3),   # negative
) -> float:
    """
    ROC-AUC for separating group_a vs. group_b.
    group_a is treated as the positive class.
    """
    yt = np.isin(y_true, group).astype(int)
    yp = np.isin(y_pred, group).astype(int)

    return roc_auc_score(yt, yp)
