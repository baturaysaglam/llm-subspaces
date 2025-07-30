# Exploring Linear Subspaces in LLM Representations

This repository contains the codebase for our paper, _Large Language Models Encode Semantics Into Low-Dimensional Linear Subspaces_.

### Summary of Key Findings

1. **Low-Dimensional Subspaces**: Large language models (LLMs) encode high-level semantics (e.g., a passage about physics) in subspaces that are significantly lower dimensional than the model’s full hidden representation space.

2. **Linear Separability Across Topics**: Hidden representations of broad scientific domains (e.g., math vs. computer science) are **linearly separable**.

3. **Not Limited to Topics**: Linearly separable structures also emerge in:
    - Representations of the *same question* with and without **chain-of-thought prompting**
    - Representations shaped by **alignment behavior**, such as responding differently to *benign* vs. *harmful* prompts with similar surface forms

4. **Practical Application**: These structured representations enable building **AI defense systems** that operate in hidden space to detect adversarial or malicious prompts—that can't be captured in token space.

## What's Inside

This repository enables you to:

1. **Extract hidden states** for any input text from a variety of LLMs.
2. **Evaluate linear separability** between representation clusters—for example, comparing domains like _physics vs. math_ or input types like _benign vs. adversarial_.
3. **Build and test** a lightweight **latent-space MLP guardrail** that classifies input prompts as **adversarial**, **benign**, or **harmful** based solely on hidden representations.


## Getting Started

We recommend creating a fresh virtual environment before installing dependencies.

### Step 1: Create a new virtual environment

```bash
python3 -m venv llm-subspaces
source llm-subspaces/bin/activate
```

### Step 2: Install required packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install `cuML`

Refer to the [RAPIDS installation guide](https://pypi.org/project/cuml/) which provides compatible versions based on your CUDA setup.

### Models

We only support pretrained or local **Hugging Face models** interfaced by the `transformers` library.


### Text Datasets

Datasets should follow the structure below:
```
datasets/
├── [DATASET_NAME]/
│   ├── [TOPIC_NAME_1].csv
│   ├── [TOPIC_NAME_2].csv
│   └── ...
```
Each `csv` file must contain a column named `text` that holds the input samples.


## 1. Collect Hidden States
Use `hidden_state.py` to pass text datasets through a specified Hugging Face model and extract hidden states from a chosen layer—specifically, the hidden representation **just before the generation of the first output token**.
```
options:
  -h, --help            show this help message and exit
  --model MODEL         The official Hugging Face model name or local path to the transformers checkpoint. The model must be causal.
  --layer LAYER         Layer number to extract hidden states from. Uses 1-based indexing, e.g., to get the 17-th layer, set this to 17.
  --dataset {arxiv,cot,wildjb}
                        Name of the dataset to be used. Datasets are assumed to be contained under './datasets/'.
  --topic {cs,eess,math,physics,q-bio,stat,  # arXiv
           commonsense_qa,commonsense_qa_cot,gsm8k,gsm8k_cot,mmlu,mmlu_cot,  # CoT
           adversarial_benign,adversarial_harmful,vanilla_benign,vanilla_harmful}  # WildJailbreak
                        Topic or subset within the dataset. For arXiv, these are the subject areas. For CoT and Wild-JB, these are the respective subsets.
  --batch_size BATCH_SIZE
                        Number of samples to process in each batch. Default is 25. Adjust based on the available GPU memory.
  --num_samples NUM_SAMPLES
                        Threshold to cap the total number of samples. If None, all samples will be used.
  --max_tokens MAX_TOKENS
                        Threshold for the maximum number of tokens. If None, the default value of 750 will be used.
  --min_tokens MIN_TOKENS
                        Threshold for the minimum number of tokens. If None, no filtering will be applied.
  --keyword_th KEYWORD_TH
                        Threshold (in percentage) up to which extent the topic-specific keywords will be masked (for sensitivity analysis).
  --mask_token MASK_TOKEN
                        The special token to use for masking the topic-specific keywords (for sensitivity analysis). Default is '<MASK>'.
```

#### **Example usage:**
```bash
python hidden_state.py \
    --model meta-llama/Llama-3.1-8B \
    --layer 32 \
    --dataset arxiv \
    --topic eess \
    --batch_size 25 \
    --num_samples 20000 \
    --max_tokens 750 \
    --min_tokens 20
```

### **Optional:** Sensitivity Analysis via Word Masking

We perform a **sensitivity analysis** in the paper (see **Section 5.3**) by masking individual words based on their frequency in English. To enable this during hidden state collection, include the following flags:
```bash
    --keyword_th 60 \
    --unigram_freq_path datasets/arxiv/unigram_freq.csv \
    --mask_token <MASK>
```
- `unigram_freq_path` should point to a CSV file containing frequency ranks for 333,333 English words.
- Running the script with a dummy path will prompt the script to download the frequency file for you.


## **2. Check Separability**

We assess linear separability by fitting a **hard-margin SVM** with parameters $C = 10^{10}$ and $\text{tol} = 10^{-12}$. If the classifier achieves **zero error** on both topic datasets in a given pair, we say the pair **linearly separable**.

We use [`cuML`](https://pypi.org/project/cuml/) for fast, GPU-accelerated SVMs.

```
options:
  -h, --help            show this help message and exit
  --model MODEL         The Hugging Face model name or path should be entered in lowercase after the first `/` in the official model name, e.g., `llama-3.1-8b`.
  --layer LAYER         Layer number to extract hidden states from. Uses 1-based indexing, e.g., to get the 17-th layer, set this to 17.
  --dataset {arxiv,cot,wildjb}
                        Name of the dataset. It has to be the same for both topics.
  --topics TOPICS [TOPICS ...]
                        List of topics. Only two topics are allowed for separability check. Example: `--topics math physics`
  --keyword_th KEYWORD_TH
                        The percentage of topic-specific keywords to mask (for sensitivity analysis). If None, no masking is applied.
  --gpu GPU             Ordinal of the GPU to use. Default is 0.
```

#### **Example usage:**
```bash
python separability.py \
    --model llama-3.1-8b-instruct \
    --layer 32 \
    --dataset wildjb \
    --topics adversarial_harmful vanilla_benign
```

To test separability under word masking (as in sensitivity analysis), simply include ``--keyword_th [KEYWORD_TH]``.

## **3. Building the Latent-Space Guardrail**

In `cookbooks/guardrail.ipynb`, you'll find a step-by-step tutorial for training and evaluating a **latent-space guardrail**—a 4-class, 7-layer MLP classifier—that identifies adversarial and harmful prompts directly in hidden space. The classifier is trained and tested on the [WildJailbreak dataset](https://huggingface.co/datasets/declare-lab/WildChat-Jailbreak).

This guardrail delivers substantial improvements in handling unsafe inputs:
- **~98%** refusal rate in latent space  
- vs. **~35%** refusal when relying only on raw text outputs

See **Section 7** of the paper for full details.
