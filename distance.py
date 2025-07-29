import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoConfig

from utils.utils import hf_model_name_dict, load_hidden_states

torch.manual_seed(42)
np.random.seed(42)
    

def sliced_wasserstein_distance(x: torch.Tensor,
                                y: torch.Tensor,
                                num_projections: int = 100,
                                device: str = 'cuda') -> float:
    """
    Compute the Sliced Wasserstein Distance between two point clouds x and y,
    subsampling the larger cloud down to the size of the smaller.
    """
    x = x.to(device)
    y = y.to(device)
    N, D = x.shape
    M, _ = y.shape

    # if sizes differ, subsample each to the smaller size k
    k = min(N, M)
    if N != k:
        idx = torch.randperm(N, device=device)[:k]
        x = x[idx]
    if M != k:
        idx = torch.randperm(M, device=device)[:k]
        y = y[idx]

    # now both are (k, D)
    # draw and normalize random directions
    dirs = torch.randn(num_projections, D, device=device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)  # (P, D)

    # project onto each direction
    proj_x = x @ dirs.t()  # (k, P)
    proj_y = y @ dirs.t()  # (k, P)

    # sort along the sample dimension
    proj_x_sorted, _ = proj_x.sort(dim=0)
    proj_y_sorted, _ = proj_y.sort(dim=0)

    # 1D Wasserstein per projection
    wass_1d = (proj_x_sorted - proj_y_sorted).abs().mean(dim=0)  # (P,)

    return wass_1d.mean().item()


def main(args):
    assert len(args.topics) == 2, "Provide exactly two topics for comparison"
    
    model_name = args.model.split('/')[-1].lower()
    model_config = AutoConfig.from_pretrained(hf_model_name_dict[model_name])

    save_path = os.path.join('distances',
                             model_name,
                             f'layer_{args.layer}')
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if os.path.join(save_path, f'{args.topics[0]}_{args.topics[1]}.json') or os.path.join(save_path, f'{args.topics[1]}_{args.topics[0]}.json'):
        print(f"\n\tRun already exists at {save_path}. Exiting to avoid overwriting...\t")

    X0 = load_hidden_states(model_name,
                            args.layer,
                            args.dataset,
                            args.topics[0],
                            embed_dim=model_config.hidden_size,
                            device=device)
    X1 = load_hidden_states(model_name,
                            args.layer,
                            args.dataset,
                            args.topics[1],
                            embed_dim=model_config.hidden_size,
                            device=device)
    
    dist = sliced_wasserstein_distance(X0, X1, num_projections=args.num_projections, device=device)

    print(f'\n---> Sliced Wasserstein Distance: {dist:.3f}\n')

    run_dict = {
        'sliced_wasserstein_dist': dist,
    }

    with open(os.path.join(save_path, f'{args.topics[0]}_{args.topics[1]}.json'), 'w') as f:
        json.dump(run_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the Sliced Wasserstein Distance between topic pairs in hidden states")
    parser.add_argument('--model',
                        type=str,
                        default='llama-3.1-8b-instruct',
                        help="Name of the model to analyze")
    parser.add_argument('--layer',
                        type=int,
                        default=32,
                        help="Layer number to analyze")
    parser.add_argument('--dataset',
                        type=str,
                        default='wildjb',
                        help="Name of the dataset to analyze")
    parser.add_argument('--topics',
                        type=str,
                        nargs='+',
                        default=['math', 'physics'],
                        help="List of topics. Only two topics are allowed for separability check. Example: `--topics math physics`")
    parser.add_argument('--num_projections',
                        type=int,
                        default=10000,
                        help="Number of projections for sliced Wasserstein distance")
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help="GPU number to use")
    args = parser.parse_args()

    main(args)
