# Standard library
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, pipeline
from vllm import LLM

from query import query_vllm_model
from utils.utils import hf_model_name_dict, load_hidden_states


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

CONTEXT_LEN = 2048

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    model = LLM(model=hf_model_name_dict[args.model],
                tensor_parallel_size=8,
                max_model_len=CONTEXT_LEN,
                max_num_seqs=1)
    tokenizer = model.get_tokenizer()
    model_conf = AutoConfig.from_pretrained(hf_model_name_dict[args.model])

    # Load the classifier trained on the model representations
    exp_dir = os.path.join('clf_models',
                           args.model,
                           f'layer_{args.layer}',
                           args.dataset.split('/eval')[0],
                           args.run_id)

    with open(os.path.join(f'{exp_dir}/args.json')) as f:
        clf_args = json.load(f)

    guardrail = MLP(
        d_in=model_conf.hidden_size,
        n_cls=4,
        hidden_size=clf_args['mlp_layers'],
        dropout_p=clf_args['dropout'],
        use_ln=clf_args['use_ln']
    ).to(device)
    guardrail.load_state_dict(torch.load(f'{exp_dir}/clf.pth', map_location=device))

    # Load Llama Guard 2
    llama_guard = pipeline("text-classification", model=args.llama_guard_model)

    prompts = pd.read_csv(f'datasets/wildjb/eval/{args.prompt_type}.csv')['text'].tolist()
    hidden_states = load_hidden_states(args.model,
                                       args.layer,
                                       dataset=args.dataset,
                                       topic=args.prompt_type,
                                       embed_dim=model_conf.hidden_size,
                                       device=device)
    
    save_path = f"model_responses/{args.model}/{args.dataset}"
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(f"{save_path}/{args.prompt_type}.json"):
        with open(f"{save_path}/{args.prompt_type}.json", "r") as f:
            responses_dict = json.load(f)
        print(f"\n\tLoaded {len(responses_dict)} existing responses from {save_path}/{args.prompt_type}.json\n")
    else:
        responses_dict = {}

    ptr = len(responses_dict)
    prompts = prompts[ptr:]
    
    dataset_p_bar = tqdm(enumerate(prompts),
                         total=len(prompts),
                         desc=f'Processing {args.prompt_type} prompts',
                         leave=True)
    for i, prompt in dataset_p_bar:
        # Get a valid response from the model
        temp = 0.0
        while True:
            response = query_vllm_model(prompt, model, tokenizer, temperature=temp).strip()

            tokenized = tokenizer.tokenize(response, add_special_tokens=False)
            is_invalid = len(tokenized) == 1 and tokenizer.convert_tokens_to_ids(tokenized[0]) in tokenizer.all_special_ids

            if not is_invalid and len(tokenized) > 3:
                break
            else:
                temp += 0.1  # if gibberish output, increase the temp by 0.1

        # Prediction of the MLP guardrail
        with torch.no_grad():
            guardrail_logits = guardrail(hidden_states[i+ptr]).cpu().numpy().flatten().tolist()
            guardrail_pred = guardrail.predict(hidden_states[i+ptr]).item()

        # BASELINE 1: Llama Guard 2
        llama_guard_pred = llama_guard(prompt)[0]
        
        responses_dict[f"sample_{i+ptr}"] = {
            'prompt': prompt,
            'type': args.prompt_type,
            'response': response,
            'final_temp': temp,
            'guardrail_clf': {
                'logits': guardrail_logits,
                'prediction': guardrail_pred
            },
            # 'llama_guard_clf': {
            #     'score': llama_guard_pred['score'],
            #     'prediction': llama_guard_pred['label']
            # }
        }

        with open(f'{save_path}/{args.prompt_type}.json', "w") as f:
            json.dump(responses_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect responses from the model.")
    parser.add_argument('--model',
                        type=str,
                        default='llama-3.1-8b-instruct',
                        help="Name of the model to analyze.")
    parser.add_argument('--layer',
                        type=int,
                        default=32,
                        help="Layer number to extract hidden states from. Uses 1-based indexing, e.g., to get the 17-th layer, set this to 17.")
    parser.add_argument('--dataset',
                        type=str,
                        default='wildjb/eval',
                        choices=[
                            'wildjb/eval',
                            'wildguardmix/eval'
                        ],
                        help="Name of the dataset. It's assumed to locate under './datasets'")
    parser.add_argument('--prompt_type',
                        type=str,
                        default='vanilla_benign',
                        help="Type of prompt dataset.")
    parser.add_argument('--run_id',
                        type=str,
                        default='988603dd',
                        help="Experiment ID for the classifier. The parameters will be automatically read from the experiment results.")
    parser.add_argument('--llama_guard_model',
                        type=str,
                        default="meta-llama/Llama-Prompt-Guard-2-86M",
                        help="Llama Guard model to use.")
    args = parser.parse_args()
 
    main(args)
