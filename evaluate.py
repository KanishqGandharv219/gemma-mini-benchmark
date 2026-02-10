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
    # Geography (15 questions)
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Italy?", "answer": "Rome"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "What is the capital of Brazil?", "answer": "Brasília"},
    {"question": "What is the capital of India?", "answer": "New Delhi"},
    {"question": "What is the capital of Russia?", "answer": "Moscow"},
    {"question": "What is the capital of Egypt?", "answer": "Cairo"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"question": "What is the smallest continent?", "answer": "Australia"},
    {"question": "How many continents are there?", "answer": "7"},
    {"question": "What is the longest river in the world?", "answer": "Nile"},
    {"question": "What is the highest mountain in the world?", "answer": "Mount Everest"},

    # Math (15 questions)
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is 5 * 6?", "answer": "30"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 144 / 12?", "answer": "12"},
    {"question": "What is the square root of 64?", "answer": "8"},
    {"question": "What is 15% of 200?", "answer": "30"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "What is the next prime number after 7?", "answer": "11"},
    {"question": "What is 2 to the power of 5?", "answer": "32"},
    {"question": "What is the value of pi rounded to 2 decimal places?", "answer": "3.14"},
    {"question": "How many sides does a hexagon have?", "answer": "6"},
    {"question": "What is 1/4 as a decimal?", "answer": "0.25"},
    {"question": "What is 20% of 50?", "answer": "10"},
    {"question": "What is the cube of 3?", "answer": "27"},
    {"question": "What is the sum of angles in a triangle?", "answer": "180"},

    # Science (20 questions)
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "What is the chemical symbol for oxygen?", "answer": "O"},
    {"question": "What is the chemical symbol for carbon?", "answer": "C"},
    {"question": "What is the speed of light in vacuum (in m/s)?", "answer": "299792458"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury"},
    {"question": "How many planets are in our solar system?", "answer": "8"},
    {"question": "What is the closest planet to the Sun?", "answer": "Mercury"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "Carbon dioxide"},
    {"question": "What gas do plants release during photosynthesis?", "answer": "Oxygen"},
    {"question": "What is the powerhouse of the cell?", "answer": "Mitochondria"},
    {"question": "What is DNA an abbreviation for?", "answer": "Deoxyribonucleic acid"},
    {"question": "What is the hardest natural substance on Earth?", "answer": "Diamond"},
    {"question": "What is the boiling point of water at sea level in Celsius?", "answer": "100"},
    {"question": "What is the freezing point of water in Celsius?", "answer": "0"},
    {"question": "How many bones are in the adult human body?", "answer": "206"},
    {"question": "What is the largest organ in the human body?", "answer": "Skin"},
    {"question": "What force keeps us on the ground?", "answer": "Gravity"},
    {"question": "What is the atomic number of hydrogen?", "answer": "1"},

    # History (15 questions)
    {"question": "What year did World War 2 end?", "answer": "1945"},
    {"question": "What year did World War 1 start?", "answer": "1914"},
    {"question": "Who was the first President of the United States?", "answer": "George Washington"},
    {"question": "In what year did the Titanic sink?", "answer": "1912"},
    {"question": "Who discovered America in 1492?", "answer": "Christopher Columbus"},
    {"question": "What year did the Berlin Wall fall?", "answer": "1989"},
    {"question": "What year did India gain independence?", "answer": "1947"},
    {"question": "Who was the first man on the moon?", "answer": "Neil Armstrong"},
    {"question": "What year did humans first land on the moon?", "answer": "1969"},
    {"question": "What ancient wonder is located in Egypt?", "answer": "Pyramids"},
    {"question": "What year did the French Revolution begin?", "answer": "1789"},
    {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
    {"question": "Who invented the light bulb?", "answer": "Thomas Edison"},
    {"question": "What year did the American Civil War end?", "answer": "1865"},
    {"question": "What ancient civilization built Machu Picchu?", "answer": "Inca"},

    # Literature (15 questions)
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
    {"question": "Who wrote 1984?", "answer": "George Orwell"},
    {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
    {"question": "Who wrote The Great Gatsby?", "answer": "F. Scott Fitzgerald"},
    {"question": "Who wrote Moby Dick?", "answer": "Herman Melville"},
    {"question": "Who wrote To Kill a Mockingbird?", "answer": "Harper Lee"},
    {"question": "Who wrote Harry Potter?", "answer": "J.K. Rowling"},
    {"question": "Who wrote The Odyssey?", "answer": "Homer"},
    {"question": "Who wrote The Iliad?", "answer": "Homer"},
    {"question": "Who wrote Macbeth?", "answer": "Shakespeare"},
    {"question": "Who wrote The Divine Comedy?", "answer": "Dante"},
    {"question": "Who wrote War and Peace?", "answer": "Leo Tolstoy"},
    {"question": "Who wrote Crime and Punishment?", "answer": "Fyodor Dostoevsky"},
    {"question": "Who wrote The Lord of the Rings?", "answer": "J.R.R. Tolkien"},

    # General Knowledge (20 questions)
    {"question": "How many days are in a leap year?", "answer": "366"},
    {"question": "How many hours are in a day?", "answer": "24"},
    {"question": "How many minutes are in an hour?", "answer": "60"},
    {"question": "How many seconds are in a minute?", "answer": "60"},
    {"question": "How many weeks are in a year?", "answer": "52"},
    {"question": "How many months are in a year?", "answer": "12"},
    {"question": "What color is the sky on a clear day?", "answer": "Blue"},
    {"question": "What is the opposite of hot?", "answer": "Cold"},
    {"question": "What animal is known as man's best friend?", "answer": "Dog"},
    {"question": "What is the largest land animal?", "answer": "Elephant"},
    {"question": "What is the fastest land animal?", "answer": "Cheetah"},
    {"question": "What bird is known for its ability to mimic human speech?", "answer": "Parrot"},
    {"question": "How many legs does a spider have?", "answer": "8"},
    {"question": "How many legs does an insect have?", "answer": "6"},
    {"question": "What is the largest mammal in the world?", "answer": "Blue whale"},
    {"question": "What is the tallest animal in the world?", "answer": "Giraffe"},
    {"question": "What do bees produce?", "answer": "Honey"},
    {"question": "What is the name of the fairy tale character who left a glass slipper?", "answer": "Cinderella"},
    {"question": "What fruit is associated with keeping doctors away?", "answer": "Apple"},
    {"question": "What vegetable makes you cry when you cut it?", "answer": "Onion"}
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
