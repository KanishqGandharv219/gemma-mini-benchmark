# Gemma Model Comparison: Factual QA Benchmark

A reproducible benchmark comparing Google's Gemma 2B and 7B models (base and instruction-tuned variants) on a 100-question factual QA dataset.

## Results

All models evaluated on 100 factual questions covering geography, math, science, history, literature, and general knowledge.

| Model | Parameters | Type | Precision | Accuracy |
|-------|-----------|------|-----------|----------|
| `google/gemma-2b` | 2B | Base | fp16 | 19.00% |
| `google/gemma-2b-it` | 2B | Instruct | fp16 | **85.00%** |
| `google/gemma-7b` | 7B | Base | fp16 | 21.00% |
| `google/gemma-7b-it` | 7B | Instruct | fp16 | 69.00% |

## Key Findings

### 1. Instruction Tuning Has Massive Impact

- **2B-IT vs 2B Base**: +66 percentage points (19% → 85%)
- **7B-IT vs 7B Base**: +48 percentage points (21% → 69%)
- Base models without instruction tuning struggle with the QA format, often generating continuations instead of direct answers

### 2. Scale Alone Doesn't Guarantee Better Performance

- **7B Base vs 2B Base**: Only +2% improvement (19% → 21%)
- Raw model size provides minimal benefit without proper instruction tuning
- For factual recall tasks, instruction alignment matters more than parameter count

### 3. Surprising Result: 2B-IT Outperforms 7B-IT

- **2B-IT: 85%** vs **7B-IT: 69%** (-16%)
- Possible explanations:
  - Memory constraints on 7B-IT required aggressive optimization (8 tokens vs 16, constant cleanup)
  - Shorter generation window may have cut off complete answers
  - 2B-IT may be better optimized for concise, factual responses
  - Instruction tuning recipe may differ between model sizes

### 4. Instruction-Tuned 2B is Production-Ready

- 85% accuracy on diverse factual questions with only 2B parameters
- 4× smaller than 7B models, faster inference, lower memory
- Strong candidate for resource-constrained deployments

## Dataset

The evaluation set consists of 100 factual questions across 6 categories:
- **Geography** (15 questions): Capitals, continents, geographical features
- **Math** (15 questions): Arithmetic, geometry, basic algebra
- **Science** (20 questions): Chemistry, physics, biology, astronomy
- **History** (15 questions): Major events, figures, dates
- **Literature** (15 questions): Authors and famous works
- **General Knowledge** (20 questions): Common facts, animals, everyday knowledge

## Methodology

- **Prompt format**: Instruction-style prompt with question-answer structure
- **Generation**: Greedy decoding (deterministic, `do_sample=False`)
- **Max tokens**: 
  - 2B models: 16 tokens per answer
  - 7B base: 16 tokens per answer
  - 7B-IT: 8 tokens per answer (memory optimization)
- **Evaluation**: Substring matching (case-insensitive, normalized)
- **Hardware**: Google Colab free tier (T4 GPU, ~15GB VRAM)

## Analysis

### Why Base Models Perform Poorly

Base models are trained on next-token prediction without explicit instruction following. When given a question, they often:
- Continue the text as if it's part of a document
- Generate multiple questions instead of answering
- Produce verbose explanations when only a short answer is needed

Example:
Question: "What is the capital of France?"
Base model: "What is the capital of France? What is the capital of Germany? ..."
IT model: "Paris"


### Why 2B-IT Outperformed 7B-IT

This counterintuitive result warrants further investigation:
1. **Memory constraints**: 7B-IT required `max_new_tokens=8` vs `16` for 2B-IT
2. **Answer completeness**: Shorter generation window may truncate multi-word answers
3. **Model optimization**: 2B-IT may be specifically tuned for concise responses
4. **Future work**: Re-run 7B-IT with 16 tokens on higher-memory hardware

## Repository Structure

gemma-benchmark/
├── README.md
├── gemma_benchmark.ipynb # Full evaluation notebook
├── eval_data.py # 100-question dataset
├── results/
│ ├── gemma2b_base_eval_summary.json
│ ├── gemma2b_base_eval_results.jsonl
│ ├── gemma2b_it_eval_summary.json
│ ├── gemma2b_it_eval_results.jsonl
│ ├── gemma7b_base_eval_summary.json
│ ├── gemma7b_base_eval_results.jsonl
│ ├── gemma7b_it_fp16_eval_summary.json
│ └── gemma7b_it_fp16_eval_results.jsonl
└── requirements.txt


## Usage

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/gemma-benchmark.git
cd gemma-benchmark
pip install -r requirements.txt
```

### 2. Set Hugging Face Token
```
import os
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
os.environ["HF_TOKEN"] = HF_TOKEN
```

Get your token from: https://huggingface.co/settings/tokens

Accept the Gemma license at: https://huggingface.co/google/gemma-2b

### 3. Run Evaluation
Open gemma_benchmark.ipynb in Google Colab and run all cells sequentially.

### Limitations
Evaluation metric: Simple substring matching may miss paraphrased correct answers

Generation mode: Greedy decoding only (no sampling or beam search)

Task scope: Limited to factual recall (no reasoning, math word problems, or open-ended generation)

Hardware constraints: 7B-IT required reduced token budget (8 vs 16) on free Colab

Dataset size: 100 questions provides initial signal but larger sets needed for statistical significance
