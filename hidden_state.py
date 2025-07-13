import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.masking import (filter_by_frequency, generate_histogram_buckets,
                           load_word_frequencies)
from utils.utils import (TextDataset, count_tokens, get_model_precision,
                         save_hidden_states)


def main(args):
    model_save_name = args.model.split('/')[-1].lower()

    # Load the data
    data_df = pd.read_csv(os.path.join('datasets',
                                       args.dataset,
                                       f'{args.topic}.csv'))

    # Remove the samples with less than the threshold token length
    if args.min_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, fast=True)
        text_samples = data_df['text'].tolist()

        token_counts = count_tokens(text_samples, tokenizer)
        indices = [i for i, x in enumerate(token_counts) if x < args.min_tokens]
        data_df = data_df.drop(indices).reset_index(drop=True)
 
    # Sample a subset with the max number of samples specified
    if args.num_samples is not None:
        data_df = data_df.sample(n=min(len(data_df), args.num_samples), random_state=42)

    # Load the model and tokenizer
    config = AutoConfig.from_pretrained(args.model)
    model_dtype = get_model_precision(config.torch_dtype)  # half-precision allows us to accommodate larger models

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 output_hidden_states=True,
                                                 torch_dtype=model_dtype,
                                                 device_map='auto')  # We shard the model using the accelerate package
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    if args.keyword_th is not None:
        assert args.keyword_th > 0 and args.keyword_th < 100, "Specify a valid percentage for keyword masking between 0 and 100."
        save_path = os.path.join('hidden_states/sensitivity_analysis',
                                 model_save_name,
                                 f'layer_{args.layer}',
                                 args.dataset,
                                 args.topic,
                                 f'{args.keyword_th}%')

        # Mask the keywords        
        log_frequencies = load_word_frequencies('datasets/arxiv/word_freq/unigram_freq.csv')  # The word frequencies assumed to be under this path
        bucket_thresholds = generate_histogram_buckets(log_frequencies, num_buckets=100)

        data_df['text'] = data_df['text'].astype(str).apply(lambda word: filter_by_frequency(word,
                                                                                             log_frequencies,
                                                                                             args.keyword_th,
                                                                                             bucket_thresholds,
                                                                                             mask_token=args.mask_token))
        # Register the masking token into the tokenizer
        new_id = tokenizer.add_tokens([args.mask_token])
        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            model.get_input_embeddings().weight[new_id].zero_()
    else:
        save_path = os.path.join('hidden_states',
                                 model_save_name,
                                 f'layer_{args.layer}',
                                 args.dataset,
                                 args.topic)

    # Load the dataset
    dataset = TextDataset(list(data_df['text']))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(save_path, exist_ok=True)
    
    p_bar = tqdm(dataloader, desc="Processing...", unit=' batch')
    with torch.no_grad():
        for batch_idx, prompt_batch in enumerate(p_bar):
            encoded_inputs = tokenizer(prompt_batch,
                                       padding=True,
                                       truncation=True,
                                       max_length=min(750, args.max_tokens) if args.max_tokens is not None else 750,
                                       return_tensors='pt').to(model.device)

            input_ids = encoded_inputs['input_ids']
            attention_mask = encoded_inputs['attention_mask']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)        
            layer_hidden_states = outputs.hidden_states[args.layer]  # tuple of shape [batch, max_seq_len, hidden_dim]

            # Taking the last token position might introduce noise because of the padding tokens
            pad_mask = (input_ids == tokenizer.pad_token_id)
            first_pad_positions = torch.where(
                pad_mask.any(dim=1),
                pad_mask.int().argmax(dim=1),
                torch.tensor(input_ids.size(1))
            ).squeeze()

            valid_positions = first_pad_positions - 1
            valid_positions = valid_positions.clamp(min=0)

            batch_indices = torch.arange(layer_hidden_states.shape[0])

            selected_hidden_states = layer_hidden_states[batch_indices, valid_positions]
            save_hidden_states(selected_hidden_states, batch_idx, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract hidden states from a causal language model for a specific layer and topic.")
    parser.add_argument('--model',
                        type=str,
                        default='meta-llama/Llama-3.1-8B',
                        help="Hugging Face model name or path. The model should be a causal language model.")
    parser.add_argument('--layer',
                        type=int,
                        default=32,
                        help="Layer number to extract hidden states from. Uses 1-based indexing, e.g., to get the 17-th layer, set this to 17.")
    parser.add_argument('--dataset',
                        type=str,
                        default='arxiv',
                        help="Name of the dataset to be used. Datasets are assumed to be contained under './datasets/")
    parser.add_argument('--topic',
                        type=str,
                        default='eess',
                        choices=[
                            # arXiv abstracts
                            'cs',
                            'eess',
                            'math',
                            'physics',
                            'q-bio',
                            'stat',
                            # Chain-of-Thought datasets
                            'commonsense_qa_cot',
                            'commonsense_qa',
                            'gsm8k_cot',
                            'gsm8k',
                            'mmlu_cot',
                            'mmlu',
                            # Alignment datasets
                            'adversarial_benign',
                            'adversarial_harmful',
                            'vanilla_benign',
                            'vanilla_harmful',
                        ])
    parser.add_argument('--batch_size',
                        type=int,
                        default=25,
                        help="Number of samples to process in each batch. Default is 25. Adjust based on the available GPU memory.")
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help="Threshold to cap the total number of samples. If None, all samples will be used.")
    parser.add_argument('--max_tokens',
                        type=int,
                        default=None,
                        help="Threshold for the maximum number of tokens. If None, the default value of 750 will be used.")
    parser.add_argument('--min_tokens',
                        type=int,
                        default=None,
                        help="Threshold for the minimum number of tokens.")
    parser.add_argument('--keyword_th',
                        type=int,
                        default=None,
                        help="Threshold (in percentage) up to which extent the topic-specific keywords will be masked (for sensitivity analysis).")
    parser.add_argument('--mask_token',
                        type=str,
                        default='<MASK>',
                        help="The special token to use for masking the topic-specific keywords. Default is '<MASK>'.")
    parser.add_argument('--gpu',
                        type=int,
                        default=0)
    args = parser.parse_args()

    main(args)
