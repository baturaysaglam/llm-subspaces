# Standard library
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

from query import init_pool, process_sample


def main(args):
    # 1) Load all the samples
    response_path = os.path.join(
        'model_responses',
        args.model,
        args.dataset,
        f'{args.prompt_type}.json'
    )
    with open(response_path, "r") as f:
        responses_dict = json.load(f)

    # 2) Freeze order and dispatch to Pool
    default_clf = 'true' if 'harmful' in args.prompt_type else 'false'
    inputs = [(i, j, default_clf) for i, j in responses_dict.items()]

    with Pool(
        processes=args.num_workers,
        initializer=init_pool,
        initargs=(args.api_version, args.gemini_model)
    ) as pool:
        results = list(tqdm(
            pool.imap(process_sample, inputs),
            total=len(inputs),
            desc=f'Processing {args.prompt_type} prompts'
        ))

    # 3) Rebuild dict in original order
    clf_responses_dict = {k: v for k, v in results}

    # 4) Write ONCE at the end
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
    
    # Gemini arguments (usually we don't touch these)
    parser.add_argument('--gemini_model',
                        type=str,
                        default='gemini-2.0-flash',
                        help="Name of the Gemini model to use for classification.")
    parser.add_argument('--api_version',
                        type=str,
                        default='v1',
                        help="API version for the GenAI client.")    
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help="Number of parallel processes to use.")
    args = parser.parse_args()

    main(args)
