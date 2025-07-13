import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)


# Let's set:
# Label 0 --> Vanilla Benign
# Label 1 --> Vanilla Harmful
# Label 2 --> Adversarial Benign
# Label 3 --> Adversarial Harmful
def evaluate_classification(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = np.arange(0, 4)
    legend = [
        'van_benign',
        'van_harmful',
        'adv_benign',
        'adv_harmful'
    ]

    acc = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Adversarial macro F1 (only for adversarial classes), handle zero divisions gracefully
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
    print(f"Adversarial Macro F1: {adv_macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels, 
        zero_division=0
    ))


def group_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_a: tuple[int, ...] = (0, 2),   # positive
    group_b: tuple[int, ...] = (1, 3),   # negative
) -> float:
    """
    ROC-AUC for separating group_a vs. group_b.
    group_a is treated as the positive class.
    """
    # P(class ∈ group_a)  –> score for positive class
    probs = F.softmax(logits, dim=-1)
    p_pos = probs[:, list(group_a)].sum(dim=1).cpu().numpy()

    # 1 if label in group_a else 0
    y_true = torch.isin(labels, torch.tensor(group_a, device=labels.device)).long()
    y_true = y_true.cpu().numpy()

    return roc_auc_score(y_true, p_pos)
