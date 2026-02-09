# Gemma Mini-Benchmark

A lightweight evaluation benchmark for Google Gemma models on factual QA tasks. Part of my GSoC 2026 preparation.

## Overview

This project evaluates Gemma-2B on a small set of factual questions covering general knowledge, basic math, and science. The goal is to establish a baseline and explore Gemma's capabilities in zero-shot question answering.

## Results

**Gemma-2B Baseline:**
- Accuracy: 50% (5/10 correct)
- Evaluation set: 10 factual QA questions
- Inference: Greedy decoding, 32 max new tokens

See `gemma_eval_summary.json` and `gemma_eval_results.jsonl` for detailed results.

## Setup

### Prerequisites

1. Accept the Gemma license at https://huggingface.co/google/gemma-2b
2. Create a Hugging Face read token at https://huggingface.co/settings/tokens
3. Set the token as an environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
