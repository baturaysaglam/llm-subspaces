import argparse
import json
import os
import uuid

import numpy as np
import torch
from transformers import AutoConfig

from utils.clf_eval import evaluate_classification
from utils.mlp import *
from utils.utils import hf_model_name_dict, load_hidden_states

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def main(args):
    args.seed = SEED
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_id = uuid.uuid4().hex[:8]
    exp_dir = os.path.join('guardrail_models',
                           run_id)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n\t↳ Logging the experiment to {run_id}\n")

    model_conf = AutoConfig.from_pretrained(hf_model_name_dict[args.model])

    topics = [
        'vanilla_benign',
        'vanilla_harmful',
        'adversarial_benign',
        'adversarial_harmful'
    ] if 'wild' in args.dataset else [topic.split('.csv') for topic in os.listdir(f'./datasets/{args.dataset}') if topic.endswith('.csv')]

    # Load the topics
    hidden_states, cls_count = {}, {}
    for i, topic in enumerate(topics):
        X = load_hidden_states(args.model,
                               args.layer,
                               args.dataset,
                               topic=topic,
                               embed_dim=model_conf.hidden_size,
                               device=device)
        hidden_states[topic] = X
        cls_count[str(i)] = X.shape[0]

    X = torch.cat([hidden_states[type] for type in topics], dim=0,).to(device)
    y = torch.tensor([i for i, type in enumerate(topics) for _ in range(hidden_states[type].shape[0])], device=device)

    clf = MLP(
        d_in=model_conf.hidden_size,
        n_cls=len(hidden_states),
        hidden_size=args.mlp_layers,
        dropout_p=args.dropout,
        use_ln=args.use_ln
    ).to(device)
    final_clf = fit(clf, X, y, args, eval_ratio=0.1, early_stop_tol=5)

    # Load the test set
    hidden_states_test, cls_count = {}, {}
    for i, topic in enumerate(topics):
        X = load_hidden_states(args.model,
                               args.layer,
                               f"{args.dataset}/eval",
                               topic=topic,
                               embed_dim=model_conf.hidden_size,
                               device=device)
        hidden_states_test[topic] = X
        cls_count[str(i)] = X.shape[0]

    X_test = torch.cat([hidden_states_test[type] for type in topics], dim=0,).to(device)
    y_test = torch.tensor([i for i, type in enumerate(topics) for _ in range(hidden_states_test[type].shape[0])], device=device)

    y_pred = final_clf.predict(X_test).detach()
    y_logits = final_clf(X_test).detach()

    test_metrics = evaluate_classification(y_true=y_test.cpu().numpy(),
                                           y_pred=y_pred.cpu().numpy(),
                                           y_logits=y_logits.cpu().numpy())

    torch.save(final_clf.state_dict(), f'{exp_dir}/clf.pth')

    with open(f'{exp_dir}/args.json', "w") as f:
        json.dump(vars(args), f, indent=4)

    with open(f'{exp_dir}/result.json', "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate linear separability between two topics in hidden space using a linear SVM.")
    # Data arguments
    parser.add_argument('--model',
                        type=str,
                        default='llama-3.1-8b-instruct',
                        help="The Hugging Face model name or path should be entered in lowercase after the first `/` in the official model name, e.g., `llama-3.1-8b`.")
    parser.add_argument('--layer',
                        type=int,
                        default=32,
                        help="Layer number to extract hidden states from. Uses 1-based indexing, e.g., to get the 17-th layer, set this to 17.")
    parser.add_argument('--dataset',
                        type=str,
                        default='wildjb',
                        choices=['arxiv',
                                 'cot',
                                 'wildjb',
                                 'wildguardmix'],
                        help="Name of the dataset. It's assumed to locate under './datasets'")
    
    # Training arguments
    parser.add_argument('--num_epochs',
                        type=int,
                        default=40,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size',
                        type=int,
                        default=2048,
                        help="Batch size for training.")
    parser.add_argument('--mlp_layers',
                        type=int,
                        nargs='+',
                        default=[2048, 2048, 512, 512, 64],
                        help="Number of layers in the MLP.")
    parser.add_argument('--lr',
                        type=float,
                        default=2.5e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help="Dropout rate for the model.")
    parser.add_argument('--use_ln',
                        action='store_true',
                        help="Use layer normalization.")

    # Misc
    parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help="Ordinal of the GPU to use. Default is 0.")
    
    args = parser.parse_args()

    main(args)