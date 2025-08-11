import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


hf_model_name_dict = {
    'foundation-sec-8b-instruct': 'fdtn-ai/Foundation-Sec-8B-Instruct',
    'mistral-small-24b-base-2501': 'mistralai/Mistral-Small-24B-Base-2501',
    'mistral-small-24b-instruct-2501': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'mistral-7b-v0.3': 'mistralai/Mistral-7B-v0.3',
    'mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B',
    'llama-3.1-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama-3.2-3b': 'meta-llama/Llama-3.2-3B',
    'llama-3.2-3b-instruct': 'meta-llama/Llama-3.2-3B-Instruct',
    'gemma-2-9b': 'google/gemma-2-9b',
    'gemma-2-2b': 'google/gemma-2-2b',
    'gpt-j-6b': 'EleutherAI/gpt-j-6b',
    'gpt2-xl': 'openai-community/gpt2-xl',
    'gpt2-large': 'openai-community/gpt2-large',
    'gpt2-medium': 'openai-community/gpt2-medium',
    'gpt2': 'openai-community/gpt2',
}


# Get model precision to save some GPU memory
def get_model_precision(config_dtype: str = None) -> torch.dtype:
    if isinstance(config_dtype, str):
        dtype_mapping = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float64': torch.float64,
        }
        config_dtype = dtype_mapping.get(config_dtype.lower(), None)
    elif config_dtype is None:
        config_dtype = torch.float32
    
    return config_dtype


class TextDataset(Dataset):
    def __init__(self, text_list: list[str]):
        self.text_list = text_list

    def __len__(self) -> int:
        return len(self.text_list)

    def __getitem__(self, idx: int) -> str:
        return self.text_list[idx]


def count_tokens(text_samples: list[str], tokenizer) -> list[int]:
    p_bar = tqdm(text_samples, total=len(text_samples), desc='Tokenizing samples...')

    token_counts = []
    for text in p_bar:
        tokenized = tokenizer.encode(text)
        token_counts.append(len(tokenized))

    return token_counts
    

def compute_basis(X: torch.Tensor, device: str = 'cuda:0') -> Tuple[np.array, np.array]:
    X = X.to(device)
    _, S, Vh = torch.linalg.svd(X - X.mean(axis=0), full_matrices=False)

    S = S.cpu().numpy()
    Vh = Vh.cpu().numpy()
    
    return S, Vh


def load_hidden_states(model_name: str,
                       layer: int,
                       dataset: str,
                       topic: str,
                       embed_dim: int,
                       keyword_th: int = None,
                       device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    if keyword_th is not None:
        hidden_path = os.path.join('hidden_states/sensitivity_analysis',
                                   model_name,
                                   f'layer_{layer}',
                                   dataset,
                                   topic,
                                   f'{keyword_th}%')
    else:
        hidden_path = os.path.join('hidden_states',
                                   model_name,
                                   f'layer_{layer}',
                                   dataset,
                                   topic)
    hidden_states = torch.ones((int(3e5), embed_dim))  # Initialize with a large random size

    ptr = 0
    for file_name in tqdm(os.listdir(hidden_path), desc='Loading tensor batches...'):
        if file_name.endswith('.pt') and 'batch' in file_name:
            hidden_batch = torch.load(os.path.join(hidden_path, file_name), map_location='cpu')
            batch_size = hidden_batch.shape[0]
            hidden_states[ptr:ptr+batch_size] = hidden_batch
            ptr += batch_size

    hidden_states = hidden_states[:ptr]
    hidden_states = hidden_states.to(dtype=torch.float32, device=device)

    return hidden_states


def save_hidden_states(hidden_states: torch.Tensor,
                       batch_idx: int,
                       save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'batch_{batch_idx}.pt')
    torch.save(hidden_states, save_path)


def prepare_binary_clf_data(X0: torch.Tensor, X1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(X0.shape[0]), torch.ones(X1.shape[0])]).reshape(-1, 1).to(X.device)
    return X, y
