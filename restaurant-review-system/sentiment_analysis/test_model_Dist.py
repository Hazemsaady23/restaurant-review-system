#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "C:/Users/Lenovo/Desktop/Complete_Model/DistilBert"


# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading model...")
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Using device: {device_name}")

clf = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=device
)

print("Model loaded successfully!\n")


# ============================================================================
# PREDEFINED TEST SAMPLES
# ============================================================================

test_samples = [
    "This restaurant was amazing, loved the service and the food!",
    "Terrible experience. The staff was rude and the food was cold.",
    "The product quality is outstanding and delivery was fast.",
    "Worst purchase ever. Completely disappointed.",
    "Average experience, nothing special but not bad either.",
    "Absolutely loved it! Will definitely come back!",
    "Waste of money. Do not recommend.",
    "Great value for the price. Highly satisfied!",
    "The customer service was horrible and unhelpful.",
    "Exceeded my expectations in every way!"
]


# ============================================================================
# BATCH TESTING
# ============================================================================

def test_predefined_samples():
    print("=" * 80)
    print("TESTING PREDEFINED SAMPLES")
    print("=" * 80)
    
    for i, text in enumerate(test_samples, 1):
        result = clf(text)[0]
        label = result['label']
        score = result['score']
        sentiment = "POSITIVE" if label == "LABEL_1" else "NEGATIVE"
        
        print(f"\n[{i}] {text}")
        print(f"    → {sentiment} (confidence: {score:.4f})")
    
    print("\n" + "=" * 80)


# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def interactive_test():
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter your text to classify (or 'quit' to exit)\n")
    
    while True:
        user_input = input("Your text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        result = clf(user_input)[0]
        label = result['label']
        score = result['score']
        sentiment = "POSITIVE" if label == "LABEL_1" else "NEGATIVE"
        
        print(f"→ {sentiment} (confidence: {score:.4f})\n")


# ============================================================================
# BATCH FILE TESTING
# ============================================================================

def test_from_file(file_path):
    print("\n" + "=" * 80)
    print(f"TESTING FROM FILE: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        for i, text in enumerate(lines, 1):
            result = clf(text)[0]
            label = result['label']
            score = result['score']
            sentiment = "POSITIVE" if label == "LABEL_1" else "NEGATIVE"
            
            print(f"\n[{i}] {text}")
            print(f"    → {sentiment} (confidence: {score:.4f})")
        
        print("\n" + "=" * 80)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

def detailed_analysis(text):
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    negative_prob = probs[0][0].item()
    positive_prob = probs[0][1].item()
    
    print(f"\nText: {text}")
    print(f"\nProbabilities:")
    print(f"  Negative: {negative_prob:.4f} ({negative_prob*100:.2f}%)")
    print(f"  Positive: {positive_prob:.4f} ({positive_prob*100:.2f}%)")
    print(f"\nPrediction: {'POSITIVE' if positive_prob > negative_prob else 'NEGATIVE'}")
    print(f"Confidence: {max(negative_prob, positive_prob):.4f}")
    print("\n" + "=" * 80)


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    while True:
        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS TESTING MENU")
        print("=" * 80)
        print("1. Test predefined samples")
        print("2. Interactive testing")
        print("3. Test from file")
        print("4. Detailed analysis (single text)")
        print("5. Exit")
        print("=" * 80)
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            test_predefined_samples()
        elif choice == '2':
            interactive_test()
        elif choice == '3':
            file_path = input("Enter file path: ").strip()
            test_from_file(file_path)
        elif choice == '4':
            text = input("Enter text for detailed analysis: ").strip()
            if text:
                detailed_analysis(text)
            else:
                print("No text entered.")
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please select 1-5.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive_test()
        elif sys.argv[1] == "--predefined":
            test_predefined_samples()
        elif sys.argv[1] == "--file" and len(sys.argv) > 2:
            test_from_file(sys.argv[2])
        else:
            print("Usage:")
            print("  python test_model.py                    # Show menu")
            print("  python test_model.py --interactive      # Interactive mode")
            print("  python test_model.py --predefined       # Test predefined samples")
            print("  python test_model.py --file <path>      # Test from file")
    else:
        main()
