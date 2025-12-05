#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "C:/Users/Lenovo/Desktop/Complete_Model/restaurant_chatbot_model"
MAX_LENGTH = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path):
    print("="*70)
    print("🤖 LOADING RESTAURANT CHATBOT")
    print("="*70)
    print(f"📁 Model path: {model_path}")
    print(f"🖥️  Device: {DEVICE}")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the model first using chatbot_training.py")
        return None, None
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    print("✅ Model loaded successfully!\n")
    print("="*70)
    
    return model, tokenizer


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_response(model, tokenizer, customer_input, 
                     temperature=0.7, top_k=50, top_p=0.95):
    
    prompt = f"Customer: {customer_input} <|endoftext|> Bot:"
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=MAX_LENGTH,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Bot:" in response:
        bot_response = response.split("Bot:")[-1].strip()
    else:
        bot_response = response.replace(prompt, "").strip()
    
    if "Customer:" in bot_response:
        bot_response = bot_response.split("Customer:")[0].strip()
    
    return bot_response


# ============================================================================
# PREDEFINED TESTS
# ============================================================================

def run_predefined_tests(model, tokenizer):
    print("\n" + "="*70)
    print("🧪 RUNNING PREDEFINED TESTS")
    print("="*70)
    
    test_cases = [
        ("The food was absolutely amazing! Best restaurant ever!", "positive"),
        ("Terrible service, cold food, will never come back.", "negative"),
        ("It was okay, nothing special really.", "neutral"),
        ("Great atmosphere but the wait was too long.", "mixed"),
        ("Perfect experience from start to finish!", "positive"),
        ("I found a hair in my food!", "negative"),
        ("What's your specialty?", "question"),
        ("Do you have vegetarian options?", "question"),
        ("Thank you for everything!", "positive"),
        ("This is taking too long.", "negative"),
        ("The pasta was delicious but the dessert was bland.", "mixed"),
        ("How late are you open?", "question"),
        ("I'd like to make a reservation.", "request"),
        ("Your staff was so friendly and helpful!", "positive"),
        ("The prices are way too high for the quality.", "negative"),
    ]
    
    for i, (review, category) in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_cases)} [{category.upper()}]")
        print(f"{'─'*70}")
        print(f"👤 Customer: {review}")
        
        response = generate_response(model, tokenizer, review)
        print(f"🤖 Bot: {response}")
    
    print(f"\n{'='*70}")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(model, tokenizer):
    print("\n" + "="*70)
    print("💬 INTERACTIVE CHATBOT MODE")
    print("="*70)
    print("Type your message and press Enter to chat with the bot.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'settings' to adjust generation parameters")
    print("  - Type 'clear' to clear conversation history")
    print("="*70 + "\n")
    
    temperature = 0.7
    top_k = 50
    top_p = 0.95
    
    conversation_history = []
    
    while True:
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for chatting! Goodbye!")
            break
        
        if user_input.lower() == 'settings':
            print("\n⚙️  Current Settings:")
            print(f"   Temperature: {temperature}")
            print(f"   Top-k: {top_k}")
            print(f"   Top-p: {top_p}")
            
            try:
                temp_input = input("   New temperature (0.1-2.0) [Enter to skip]: ").strip()
                if temp_input:
                    temperature = max(0.1, min(2.0, float(temp_input)))
                
                k_input = input("   New top-k (1-100) [Enter to skip]: ").strip()
                if k_input:
                    top_k = max(1, min(100, int(k_input)))
                
                p_input = input("   New top-p (0.1-1.0) [Enter to skip]: ").strip()
                if p_input:
                    top_p = max(0.1, min(1.0, float(p_input)))
                
                print(f"\n✅ Settings updated!")
            except ValueError:
                print("❌ Invalid input. Settings unchanged.")
            
            continue
        
        if user_input.lower() == 'clear':
            conversation_history = []
            print("🧹 Conversation history cleared!\n")
            continue
        
        if not user_input:
            print("Please type a message.\n")
            continue
        
        response = generate_response(model, tokenizer, user_input, 
                                    temperature=temperature, 
                                    top_k=top_k, 
                                    top_p=top_p)
        
        print(f"🤖 Bot: {response}\n")
        
        conversation_history.append({
            'customer': user_input,
            'bot': response
        })


# ============================================================================
# BATCH TESTING FROM FILE
# ============================================================================

def batch_test_from_file(model, tokenizer, file_path):
    print("\n" + "="*70)
    print(f"📄 BATCH TESTING FROM FILE: {file_path}")
    print("="*70)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"📊 Found {len(lines)} test cases\n")
        
        for i, line in enumerate(lines, 1):
            print(f"\n{'─'*70}")
            print(f"Test {i}/{len(lines)}")
            print(f"{'─'*70}")
            print(f"👤 Customer: {line}")
            
            response = generate_response(model, tokenizer, line)
            print(f"🤖 Bot: {response}")
        
        print(f"\n{'='*70}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")


# ============================================================================
# SENTIMENT-BASED TESTING
# ============================================================================

def sentiment_based_testing(model, tokenizer):
    print("\n" + "="*70)
    print("😊 SENTIMENT-BASED RESPONSE TESTING")
    print("="*70)
    
    test_sentiments = {
        "Positive": [
            "This place is incredible! The food, service, everything!",
            "Best dining experience I've ever had!",
            "The chef deserves an award for this amazing meal!",
        ],
        "Negative": [
            "Worst meal I've ever had. Total waste of money.",
            "The service was rude and the food was terrible.",
            "I'll never eat here again. Completely disappointed.",
        ],
        "Neutral": [
            "It was fine. Nothing stood out particularly.",
            "Average food, average service, average price.",
            "Not bad, but not great either.",
        ],
        "Questions": [
            "What time do you close?",
            "Do you have gluten-free options?",
            "Can I see the menu?",
        ]
    }
    
    for sentiment, reviews in test_sentiments.items():
        print(f"\n{'='*70}")
        print(f"📊 {sentiment.upper()} REVIEWS")
        print(f"{'='*70}")
        
        for i, review in enumerate(reviews, 1):
            print(f"\n{'─'*70}")
            print(f"{sentiment} Test {i}/{len(reviews)}")
            print(f"{'─'*70}")
            print(f"👤 Customer: {review}")
            
            response = generate_response(model, tokenizer, review)
            print(f"🤖 Bot: {response}")


# ============================================================================
# COMPARISON MODE
# ============================================================================

def comparison_mode(model, tokenizer):
    print("\n" + "="*70)
    print("🔄 COMPARISON MODE - Different Temperature Settings")
    print("="*70)
    
    test_input = input("\n👤 Enter a customer review: ").strip()
    
    if not test_input:
        test_input = "The food was great but the service was slow."
    
    temperatures = [0.3, 0.7, 1.0, 1.3]
    
    print(f"\n{'='*70}")
    print(f"Testing: {test_input}")
    print(f"{'='*70}")
    
    for temp in temperatures:
        print(f"\n{'─'*70}")
        print(f"Temperature: {temp}")
        print(f"{'─'*70}")
        
        response = generate_response(model, tokenizer, test_input, temperature=temp)
        print(f"🤖 Response: {response}")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    model, tokenizer = load_model(MODEL_PATH)
    
    if model is None:
        return
    
    while True:
        print("\n" + "="*70)
        print("🍽️  RESTAURANT CHATBOT TESTING MENU")
        print("="*70)
        print("1. Run predefined tests")
        print("2. Interactive chat mode")
        print("3. Batch test from file")
        print("4. Sentiment-based testing")
        print("5. Comparison mode (different temperatures)")
        print("6. Single test")
        print("7. Exit")
        print("="*70)
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == '1':
            run_predefined_tests(model, tokenizer)
        
        elif choice == '2':
            interactive_mode(model, tokenizer)
        
        elif choice == '3':
            file_path = input("Enter file path: ").strip()
            batch_test_from_file(model, tokenizer, file_path)
        
        elif choice == '4':
            sentiment_based_testing(model, tokenizer)
        
        elif choice == '5':
            comparison_mode(model, tokenizer)
        
        elif choice == '6':
            review = input("Enter customer review: ").strip()
            if review:
                print(f"\n👤 Customer: {review}")
                response = generate_response(model, tokenizer, review)
                print(f"🤖 Bot: {response}")
            else:
                print("No input provided.")
        
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-7.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model, tokenizer = load_model(MODEL_PATH)
        if model is not None:
            if sys.argv[1] == "--test":
                run_predefined_tests(model, tokenizer)
            elif sys.argv[1] == "--interactive":
                interactive_mode(model, tokenizer)
            elif sys.argv[1] == "--file" and len(sys.argv) > 2:
                batch_test_from_file(model, tokenizer, sys.argv[2])
            elif sys.argv[1] == "--sentiment":
                sentiment_based_testing(model, tokenizer)
            else:
                print("Usage:")
                print("  python test_chatbot.py                    # Show menu")
                print("  python test_chatbot.py --test             # Run predefined tests")
                print("  python test_chatbot.py --interactive      # Interactive mode")
                print("  python test_chatbot.py --file <path>      # Batch test from file")
                print("  python test_chatbot.py --sentiment        # Sentiment-based tests")
    else:
        main()
