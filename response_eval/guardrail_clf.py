# Standard library
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tqdm import tqdm
from transformers import AutoConfig

from utils.mlp import MLP
from utils.utils import hf_model_name_dict, load_hidden_states


def main(model_name,
         run_id,
         dataset,
         prompt_type,
         gpu_id):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Load the classifier configuration trained on the model representations
    with open(os.path.join(f'guardrail_models/{run_id}/args.json')) as f:
        clf_args = json.load(f)

    assert clf_args['dataset'] == dataset.split('/eval')[0], f"Dataset mismatch detected between classifier and current dataset."\
        f"\n\tDataset: {dataset}, Classifier: {clf_args['dataset']}"

    model_conf = AutoConfig.from_pretrained(hf_model_name_dict[clf_args['model']])
    hidden_states = load_hidden_states(model_name,
                                       clf_args['layer'],
                                       dataset=dataset,
                                       topic=prompt_type,
                                       embed_dim=model_conf.hidden_size,
                                       device=device)
    
    # Load the collected response JSON
    response_path = os.path.join(
        'model_responses',
        model_name,
        dataset,
        f'{prompt_type}.json'
    )
    # Check if the model responses particularly exist fo layer
    with open(response_path, "r") as f:
        responses_dict = json.load(f)

    if clf_args['model'] != model_name:
        Warning(f"Model mismatch detected between classifier and current model.\n\tModel: {model_name}, Classifier: {clf_args['model']}")

    # Load the classifier
    guardrail = MLP(
        d_in=model_conf.hidden_size,
        n_cls=4,
        hidden_size=clf_args['mlp_layers'],
        dropout_p=clf_args['dropout'],
        use_ln=clf_args['use_ln']
    ).to(device)
    guardrail.load_state_dict(torch.load(f'guardrail_models/{run_id}/clf.pth', map_location=device))

    with torch.no_grad():
        guardrail_logits = guardrail(hidden_states)
        guardrail_pred = guardrail_logits.argmax(dim=-1)

    clf_responses_dict = {}
    p_bar = tqdm(
        zip(responses_dict.items(), guardrail_logits, guardrail_pred),
        total=len(responses_dict),     
        desc="Classifying responses",
    )
    
    for (key, value), logits, pred in p_bar:
        entry = value
        clf_responses_dict[key] = entry

        if f"guardrail-{clf_args['model']}_clf" not in clf_responses_dict[key]:
            clf_responses_dict[key][f"guardrail-{clf_args['model']}_clf"] = {}

        clf_responses_dict[key][f"guardrail-{clf_args['model']}_clf"][f"layer_{clf_args['layer']}-{args.run_id}"] = {
            'logits': logits.cpu().numpy().tolist(),
            'prediction': pred.item()
        }

    with open(response_path, "w") as f:
        json.dump(clf_responses_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classify the model responses based on hidden representations using guardrail." \
        "It overrides and adds the guardrail's predictions to the existing response JSON."
    )
    # Data arguments
    # We keep this as a separate arg so that guardrails trained on other models' representations can be tested on other models
    parser.add_argument('--model',
                        type=str,
                        default='llama-3.1-8b-instruct',
                        help="The model which we classify its responses." \
                        " The Hugging Face model name or path should be entered in lowercase after the first `/` in the official model name, e.g., `llama-3.1-8b`.")
    parser.add_argument('--dataset',
                        type=str,
                        default='wildjb/eval',
                        choices=['wildjb/eval',
                                 'wildguardmix/eval'],
                        help="Name of the dataset. It's assumed to locate under './datasets'")
    parser.add_argument('--prompt_types',
                        type=str,
                        nargs='+',
                        default=[
                            'vanilla_benign',
                            'vanilla_harmful',
                            'adversarial_benign',
                            'adversarial_harmful',
                        ],
                        help="Type(s) of prompt dataset. Provide one or more types separated by spaces.")

    # Guardrail arguments (usually we don't touch these)
    parser.add_argument('--run_id',
                        type=str,
                        default='a322fd24',  # Llama-3.1-8B-Instruct: a322fd24; Foundation-Sec-8B-Instruct: 68b2ac73 
                        help="Experiment ID for the classifier. The parameters will be automatically read from the experiment results.")
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help="Ordinal of the GPU to use. Default is 0.")
    args = parser.parse_args()

    p_bar = tqdm(args.prompt_types,
                 total=len(args.prompt_types),
                 desc='Processing prompt types')

    for prompt_type in args.prompt_types:
        main(model_name=args.model,
             dataset=args.dataset,
             prompt_type=prompt_type,
             run_id=args.run_id,
             gpu_id=args.gpu)
