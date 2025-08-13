import os
from typing import Tuple

import torch
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


def prepare_binary_clf_data(X0: torch.Tensor, X1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(X0.shape[0]), torch.ones(X1.shape[0])]).reshape(-1, 1).to(X.device)
    return X, y
