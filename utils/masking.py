import re

import numpy as np
import pandas as pd


def load_word_frequencies(file_path: str ='unigram_freq.csv') -> dict:
    """
    Load word frequencies from the Kaggle dataset.
    Expected CSV format: word,count
    """
    try:
        df = pd.read_csv(file_path)

        # Convert to dictionary for faster lookup
        word_freqs = dict(zip(df['word'].str.lower(), np.log10(df['count'] + 1)))
        
        return word_freqs
    except FileNotFoundError:
        print("The word frequencies are expected to locate under 'datasets/arxiv/word_freq/unigram_freq.csv'.")
        print("Please download the dataset from Kaggle and save as 'datasets/arxiv/word_freq/unigram_freq.csv'")
        print("Dataset: https://www.kaggle.com/datasets/rtatman/english-word-frequency")

        return None


def generate_histogram_buckets(word_frequencies: dict, num_buckets: int = 100) -> np.ndarray:
    """
    Generate histogram buckets for word frequencies.
    """
    # Get frequency values and convert to log scale since frequencies vary widely
    log_frequencies = np.array(list(word_frequencies.values()))
    
    # Create histogram buckets
    _, bin_edges = np.histogram(log_frequencies, bins=num_buckets)
    
    return bin_edges    


def filter_by_frequency(text: str,
                        log_frequencies: dict,
                        threshold: int,
                        bucket_thresholds: list,
                        mask_token: str = '<MASK>') -> str:
    """
    Filter words based on their frequency in the English language.
    
    Args:
        text (str): Input text
        log_frequencies (dict): Dictionary of word frequencies from Kaggle dataset
        bucket_thresholds (list): List of bucket thresholds
    
    Returns:
        tuple: (filtered text, removed words)
    """
    # Normalize text and split into words
    # Track word spans in original text
    text_lower = text.lower()
    
    # Use regex to find words while preserving their original position
    replacement_spans = []
    matches = []
    for match in re.finditer(r'\b\w+\b', text_lower):
        matches.append(match.group())
        if match.group().lower() not in log_frequencies or log_frequencies[match.group().lower()] < bucket_thresholds[threshold]:
            replacement_spans.append({
                'start': match.start(),
                'end': match.end(),
                'length': match.end() - match.start(),
            })
        
    for span in replacement_spans[::-1]:
        text = text[:span['start']] + mask_token + text[span['end']:]
    
    return text
