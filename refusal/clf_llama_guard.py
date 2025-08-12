# Standard library
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from vllm import LLM

from query import query_llama_guard


def main(args):
    response_path = os.path.join(
        'model_responses',
        args.model,
        args.dataset,
        f'{args.prompt_type}.json'
    )
    with open(response_path, "r") as f:
        responses_dict = json.load(f)

    model = LLM(model=args.llama_guard_model,
            tensor_parallel_size=8,
            max_num_seqs=1)
    tokenizer = model.get_tokenizer()

    clf_responses_dict = {}
    p_bar = tqdm(responses_dict.items(),
             total=len(responses_dict),     
             desc="Classifying responses",)
    for key, value in p_bar:
        entry = value
        entry['llama_guard_clf'] = {}
        entry['llama_guard_clf']['prediction'] = query_llama_guard(entry['prompt'], model, tokenizer)
        clf_responses_dict[key] = entry

    with open(response_path, "w") as f:
        json.dump(clf_responses_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classify the model responses as 'refusal' using Gemini API."
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
                        choices=['wildjb/eval',
                                 'wildguardmix/eval'],
                        help="Name of the dataset. It's assumed to locate under './datasets'")
    parser.add_argument('--prompt_type',
                        type=str,
                        default='vanilla_benign',
                        help="Type of prompt dataset.")

    # Llama Guard arguments (usually we don't touch these)
    parser.add_argument('--llama_guard_model',
                        type=str,
                        default='meta-llama/Llama-Guard-3-8B',
                        help="Full Hugging Face name of the Llama Guard model to use for classification.")
    args = parser.parse_args()

    main(args)
