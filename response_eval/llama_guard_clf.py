# Standard library
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from vllm import LLM

from utils.query import query_vllm_model

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def main(model,
         tokenizer,
         dataset,
         prompt_type,
         llama_guard_model,):
    llama_guard_model_name = llama_guard_model.llm_engine.model_config.model
    llama_guard_model_name = llama_guard_model_name.split('/')[-1].lower()
    
    # Load the collected response JSON
    response_path = os.path.join(
        'model_responses',
        model,
        dataset,
        f'{prompt_type}.json'
    )
    with open(response_path, "r") as f:
        responses_dict = json.load(f)

    clf_responses_dict = {}
    p_bar = tqdm(responses_dict.items(),
             total=len(responses_dict),     
             desc=f"Classifying responses to {prompt_type}")

    for key, value in p_bar:
        entry = value

        prediction = query_vllm_model(entry['prompt'],
                                      llama_guard_model,
                                      tokenizer,
                                      max_tokens=250,
                                      temperature=0.0)

        entry['llama-guard_clf'] = {}
        entry['llama-guard_clf'][llama_guard_model_name] = {}
        entry['llama-guard_clf'][llama_guard_model_name]['prediction'] = prediction

        clf_responses_dict[key] = entry

    with open(response_path, "w") as f:
        json.dump(clf_responses_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classify the model responses as 'refusal' using Gemini API." \
        "It overrides and adds Llama Guard's predictions to the existing response JSON."
    )
    # Data arguments
    parser.add_argument('--model',
                        type=str,
                        default='llama-3.1-8b-instruct',
                        help="The model which we classify its responses." \
                        " The Hugging Face model name or path should be entered in lowercase after the first `/` in the official model name, e.g., `llama-3.1-8b`.")
    parser.add_argument('--dataset',
                        type=str,
                        default='wildjb/eval',
                        choices=[
                            'wildjb/eval',
                            'wildguardmix/eval',
                            'harmbench'
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

    # Llama Guard arguments (usually we don't touch these)
    parser.add_argument('--llama_guard_model',
                        type=str,
                        default='meta-llama/Llama-Guard-3-8B',
                        choices=[
                            'meta-llama/Llama-Guard-3-8B',
                            'meta-llama/Llama-Guard-3-1B',
                        ],
                        help="Full Hugging Face name of the Llama Guard model to use for classification.")
    args = parser.parse_args()

    llama_guard_model = LLM(model=args.llama_guard_model,
                            tensor_parallel_size=8,
                            max_num_seqs=1)
    tokenizer = llama_guard_model.get_tokenizer()

    p_bar = tqdm(args.prompt_types,
                 total=len(args.prompt_types),
                 desc='Processing prompt types')
    
    for prompt_type in p_bar:
        main(model=args.model,
             tokenizer=tokenizer,
             dataset=args.dataset,
             prompt_type=prompt_type,
             llama_guard_model=llama_guard_model,)
