"""
Gemma Mini-Benchmark: Factual QA Evaluation
GSoC 2026 preparation project
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME = "google/gemma-2b"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Evaluation dataset
EVAL_DATA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "What year did World War 2 end?", "answer": "1945"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "How many continents are there?", "answer": "7"},
    {"question": "What is the speed of light in vacuum (in m/s)?", "answer": "299792458"},
]

def load_model():
    """Load Gemma model and tokenizer"""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("Model loaded successfully!")
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    """Generate answer from model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate(model, tokenizer, eval_data):
    """Run evaluation and return results"""
    correct = 0
    results = []
    
    print(f"\nEvaluating on {len(eval_data)} questions...\n")
    
    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        gold_answer = item["answer"]
        pred = generate_answer(model, tokenizer, question)
        is_correct = gold_answer.lower() in pred.lower()
        correct += int(is_correct)
        
        result = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted": pred,
            "correct": is_correct
        }
        results.append(result)
        
        print(f"[{i}/{len(eval_data)}] {'✓' if is_correct else '✗'} {question}")
    
    accuracy = correct / len(eval_data)
    return results, accuracy, correct

def save_results(results, accuracy, correct, total):
    """Save evaluation results to files"""
    with open("gemma_eval_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    summary = {
        "model": MODEL_NAME,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy
    }
    
    with open("gemma_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Final Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"{'='*60}")
    print("\nResults saved to:")
    print("  - gemma_eval_results.jsonl")
    print("  - gemma_eval_summary.json")

if __name__ == "__main__":
    model, tokenizer = load_model()
    results, accuracy, correct = evaluate(model, tokenizer, EVAL_DATA)
    save_results(results, accuracy, correct, len(EVAL_DATA))
