import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)


def evaluate_classification(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_logits: np.ndarray = None) -> dict:
    """
    Computes:
        - Overall performance: Accuracy, Balanced Accuracy, Macro F1 (due to class imbalance)
        - Harmful vs. Benign: Macro F1 (due to class imbalance), ROC-AUC over {0, 2} vs. {1, 3} (differentiating harmful vs. benign)
        - Adversarial vs. Vanilla: Accuracy (classes are balanced), F1 (for tradeoff between precision and recall)

    Classes:
        - 0: Vanilla benign -> 1000 samples (augmented)
        - 1: Vanilla harmful -> 1000 samples (augmented)
        - 2: Adversarial benign -> 210 samples
        - 3: Adversarial harmful -> 2000 samples
    """
    n_cls = len(np.unique(y_true))
    labels = np.arange(0, n_cls)

    # Overall performance
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1_clf = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Harmful vs. Benign
    # Mapping from original labels -> super-classes: harmful vs. benign
    harmful_map_arr = np.array([0, 1, 0, 1])

    y_test_bin = harmful_map_arr[y_true]
    y_pred_bin = harmful_map_arr[y_pred]

    macro_f1_harmful = f1_score(y_test_bin, y_pred_bin, average='macro', zero_division=0)

    if y_logits is not None:
        roc_auc = group_auc(y_true, y_logits, pos_cls=(1, 3))
    else:
        roc_auc = None

    # Adversarial vs. Vanilla
    adv_map_arr = np.array([0, 0, 1, 1])

    y_test_bin = adv_map_arr[y_true]
    y_pred_bin = adv_map_arr[y_pred]

    acc_adv = accuracy_score(y_test_bin, y_pred_bin)
    f1_adv = f1_score(y_test_bin, y_pred_bin, average='binary', zero_division=0)
    
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Class 0 --> vanilla benign")
    print("Class 1 --> vanilla harmful")
    print("Class 2 --> adversarial benign")
    print("Class 3 --> adversarial harmful\n")
    
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels), "\n")

    print("Harmful vs. Benign:")
    print(f"↳ Harmful Macro F1: {macro_f1_harmful:.4f}")

    if roc_auc is not None:
        print(f"↳ Harmful ROC-AUC: {roc_auc:.4f}\n")

    print("Adversarial vs. Vanilla:")
    print(f"↳ Adversarial Accuracy: {acc_adv:.4f}")
    print(f"↳ Adversarial F1: {f1_adv:.4f}\n")

    print("Classification Report:")
    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels, 
        zero_division=0
    ))

    return {
        'classification': {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1_clf,
            'confusion_matrix': cm.tolist()
        },
        'benign-harmful': {
            'macro_f1': macro_f1_harmful,
            'roc_auc': roc_auc
        },
        'vanilla-adversarial': {
            'accuracy': acc_adv,
            'macro_f1': f1_adv
        }
    }


def group_auc(
        y_true: np.ndarray,
        logits: np.ndarray,
        pos_cls: tuple[int] = (1, 3)  # Treat harmful as the positive class
    ) -> float:
    """
    ROC-AUC for separating group_a vs. group_b.
    group_a is treated as the positive class.
    """
    # Collapse labels: positive = 1 if class in {1, 3}, else 0
    y_binary = np.isin(y_true, pos_cls).astype(int)

    # Collapse predicted probabilities: p_pos = p(class 1) + p(class 3)
    p_pos = logits[:, pos_cls[0]] + logits[:, pos_cls[1]]

    return roc_auc_score(y_binary, p_pos)
