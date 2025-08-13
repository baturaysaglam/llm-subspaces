# Standard library
import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM

from utils.query import query_vllm_model

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

CONTEXT_LEN = 2048

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(model,
         tokenizer,
         dataset,
         prompt_type):
    model_name = model.llm_engine.model_config.model
    prompts = pd.read_csv(f'datasets/wildjb/eval/{prompt_type}.csv')['text'].tolist()

    save_path = f"model_responses/{model_name}/{dataset}"
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(f"{save_path}/{prompt_type}.json"):
        with open(f"{save_path}/{prompt_type}.json", "r") as f:
            responses_dict = json.load(f)
        print(f"\n\tLoaded {len(responses_dict)} existing responses from {save_path}/{prompt_type}.json\n")
    else:
        responses_dict = {}

    ptr = len(responses_dict)
    prompts = prompts[ptr:]
    
    dataset_p_bar = tqdm(enumerate(prompts),
                         total=len(prompts),
                         desc=f'Processing {prompt_type} prompts',
                         leave=True)
    
    for i, prompt in dataset_p_bar:
        # Get a valid response from the model
        temp = 0.0
        while True:
            response = query_vllm_model(prompt,
                                        model,
                                        tokenizer,
                                        max_tokens=512,
                                        temperature=temp)

            tokenized = tokenizer.tokenize(response, add_special_tokens=False)
            is_invalid = len(tokenized) == 1 and tokenizer.convert_tokens_to_ids(tokenized[0]) in tokenizer.all_special_ids

            if not is_invalid and len(tokenized) > 3:
                break
            else:
                temp += 0.1  # If gibberish output, increase the temperature by 0.1

        responses_dict[f"sample_{i+ptr+1}"] = {
            'prompt': prompt,
            'type': prompt_type,
            'response': response,
            'final_temp': temp,
        }

        with open(f'{save_path}/{prompt_type}.json', "w") as f:
            json.dump(responses_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect responses from an HF model.")
    parser.add_argument('--model',
                        type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help="The official Hugging Face model name or local path to the transformers checkpoint. The model must be causal.")
    parser.add_argument('--dataset',
                        type=str,
                        default='wildjb/eval',
                        choices=[
                            'wildjb/eval',
                            'wildguardmix/eval'
                        ],
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
    args = parser.parse_args()

    model = LLM(model=args.model,
                tensor_parallel_size=8,
                max_model_len=CONTEXT_LEN,
                max_num_seqs=1)

    p_bar = tqdm(args.prompt_types,
                 total=len(args.prompt_types),
                 desc='Processing prompt types')
    
    for prompt_type in p_bar:
        main(model=model,
             tokenizer=model.get_tokenizer(),
             dataset=args.dataset,
             prompt_type=prompt_type)
