from typing import Tuple

import numpy as np
import torch


def compute_basis(X: torch.Tensor, device: str = "cuda:0") -> Tuple[np.array, np.array]:
    X = X.to(device)
    _, S, Vh = torch.linalg.svd(X - X.mean(axis=0), full_matrices=False)

    S = S.cpu().numpy()
    Vh = Vh.cpu().numpy()
    
    return S, Vh
