import argparse
import json
import os

import numpy as np
import torch
from cuml.svm import LinearSVC
from transformers import AutoConfig

from utils.utils import (hf_model_name_dict, load_hidden_states,
                         prepare_binary_clf_data)

torch.manual_seed(42)
np.random.seed(42)


def fit_svm(X: torch.Tensor,
            y: torch.Tensor,
            C: int,
            tol: float,
            max_iter: int) -> tuple[LinearSVC, float]:
    clf = LinearSVC(C=C, tol=tol, max_iter=max_iter)

    clf.fit(X, y)
    accuracy = clf.score(X, y)
    
    return clf, accuracy


def main(args):
    assert len(args.topics) == 2, "Provide exactly two topics for comparison"

    if args.keyword_th is not None:
        assert  args.keyword_th > 0 and args.keyword_th < 100, "Specify a valid percentage between 0 and 100 to mask the keywords"

    model_name = args.model.split('/')[-1].lower()
    run_dir = os.path.join('runs',
                           model_name,
                           f'layer_{args.layer}',
                           args.dataset,
                           f'{args.keyword_th}%') if args.keyword_th is not None else os.path.join('runs',
                                                                                                   model_name,
                                                                                                   f'layer_{args.layer}',
                                                                                                   args.dataset)
    os.makedirs(run_dir, exist_ok=True)
    save_path = os.path.join(run_dir, f'{args.topics[0]}_{args.topics[1]}.json')

    if os.path.exists(save_path) or os.path.join(run_dir, f'{args.topics[1]}_{args.topics[0]}.json'):
        exit(f"Run already exists at {save_path}. Exiting to avoid overwriting...")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model_config = AutoConfig.from_pretrained(hf_model_name_dict[model_name])

    X0 = load_hidden_states(model_name,
                            args.layer,
                            args.dataset,
                            args.topics[0],
                            embed_dim=model_config.hidden_size,
                            keyword_th=args.keyword_th,
                            device=device)
    X1 = load_hidden_states(model_name,
                            args.layer,
                            args.dataset,
                            args.topics[1],
                            embed_dim=model_config.hidden_size,
                            keyword_th=args.keyword_th,
                            device=device)
    X, y = prepare_binary_clf_data(X0, X1)

    run_dict = {}

    svm_clf, accuracy = fit_svm(X,
                                y,
                                C=1e10,
                                tol=1e-12,
                                max_iter=int(1e9))
    
    # Compute the number of mistakes per class
    n_mistakes_cls_0 = (svm_clf.predict(X0) != 0).sum().item()
    n_mistakes_cls_1 = (svm_clf.predict(X1) != 1).sum().item()

    # Explicitly mark the perfect accuracy 1.0 only if there are no mistakes
    accuracy = 1.0 if n_mistakes_cls_0 + n_mistakes_cls_1 == 0 else accuracy

    print(f'\n---> SVM Accuracy: {accuracy:.8f}\n')
    print(f"\tNumber of misclassifications in class 0: {n_mistakes_cls_0}")
    print(f"\tNumber of misclassifications in class 1: {n_mistakes_cls_1}\n\n")

    run_dict["svm"] = {
        'accuracy': accuracy,
        'n_mistakes_cls_0': n_mistakes_cls_0,
        'n_mistakes_cls_1': n_mistakes_cls_1
    }

    # Save the results
    with open(save_path, 'w') as f:
        json.dump(run_dict, f, indent=4, default=lambda x: format(x, '.8f'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate linear separability between two topics in hidden space using a linear SVM.")
    parser.add_argument('--model',
                        type=str,
                        default='llama-3.1-8b',
                        help="The Hugging Face model name or path should be entered in lowercase after the first `/` in the official model name, e.g., `llama-3.1-8b`.")
    parser.add_argument('--layer',
                        type=int,
                        default=32,
                        help="Layer number to extract hidden states from. Uses 1-based indexing, e.g., to get the 17-th layer, set this to 17.")
    parser.add_argument('--dataset',
                        type=str,
                        default='arxiv',
                        choices=['arxiv', 'cot', 'wildjb'],
                        help="Name of the dataset. It has to be the same for both topics.")
    parser.add_argument('--topics',
                        type=str,
                        nargs='+',
                        help="List of topics. Only two topics are allowed for separability check. Example: `--topics math physics`")
    parser.add_argument('--keyword_th',
                        type=int,
                        default=None,
                        help="The percentage of topic-specific keywords to mask (for sensitivity analysis). If None, no masking is applied.")
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help="Ordinal of the GPU to use. Default is 0.")
    args = parser.parse_args()

    main(args)
