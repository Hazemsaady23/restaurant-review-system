#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup
)
import pandas as pd
from tqdm import tqdm
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "distilgpt2"
DATASET_FILE = "combined_training_data.csv"
OUTPUT_DIR = "./restaurant_chatbot_model"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("🎓 RESTAURANT CHATBOT TRAINING")
print("="*70)
print(f"🖥️  Device: {DEVICE}")
print(f"📊 Epochs: {EPOCHS}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"📚 Max Length: {MAX_LENGTH}")
print(f"🎯 Learning Rate: {LEARNING_RATE}")
print("="*70)


# ============================================================================
# DATASET CLASS
# ============================================================================

class RestaurantChatDataset(Dataset):
    
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"\n📥 Loading dataset from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(self.df)} examples")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total examples: {len(self.df)}")
        print(f"\n   Sentiment Distribution:")
        sentiment_counts = self.df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   - {sentiment}: {count} ({percentage:.1f}%)")
        
        self.examples = []
        print(f"\n🔄 Preparing training examples...")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            review = str(row['review']).strip()
            response = str(row['response']).strip()
            
            text = f"Customer: {review} <|endoftext|> Bot: {response} <|endoftext|>"
            self.examples.append(text)
        
        print(f"✅ Prepared {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, tokenizer, epochs=3):
    
    model.to(DEVICE)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    print(f"\n{'='*70}")
    print("🚀 STARTING TRAINING")
    print(f"{'='*70}")
    print(f"   Total batches per epoch: {len(train_loader)}")
    print(f"   Total training steps: {total_steps}")
    print(f"   Warmup steps: 100")
    print(f"{'='*70}\n")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*70}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch + 1} Complete!")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"   🌟 New best loss! (Previous: {best_loss:.4f})")
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"{'='*70}")


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_model(model, tokenizer):
    
    print("\n" + "="*70)
    print("🧪 TESTING THE TRAINED MODEL")
    print("="*70)
    
    model.eval()
    model.to(DEVICE)
    
    test_reviews = [
        "The food was absolutely amazing! Best restaurant ever!",
        "Terrible service, cold food, will never come back.",
        "It was okay, nothing special really.",
        "Great atmosphere but the wait was too long.",
        "Perfect experience from start to finish!",
        "I found a hair in my food!",
        "What's your specialty?",
        "Do you have vegetarian options?",
        "Thank you for everything!",
        "This is taking too long.",
    ]
    
    for i, review in enumerate(test_reviews, 1):
        prompt = f"Customer: {review} <|endoftext|> Bot:"
        
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Bot:" in response:
            bot_response = response.split("Bot:")[-1].strip()
        else:
            bot_response = response.replace(prompt, "").strip()
        
        print(f"\n{'─'*70}")
        print(f"Test {i}/10")
        print(f"{'─'*70}")
        print(f"👤 Customer: {review}")
        print(f"🤖 Bot: {bot_response}")
    
    print(f"\n{'='*70}")


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model, tokenizer, output_dir):
    print(f"\n{'='*70}")
    print(f"💾 SAVING MODEL")
    print(f"{'='*70}")
    print(f"   Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"   ✅ Model saved successfully!")
    print(f"   📦 Model size: {size_mb:.2f} MB")
    print(f"{'='*70}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎓 RESTAURANT CHATBOT TRAINING PIPELINE")
    print("="*70)
    
    if not os.path.exists(DATASET_FILE):
        print(f"\n❌ Dataset file not found: {DATASET_FILE}")
        print("Please run 'download_datasets.py' first!")
        return
    
    print("\n📦 Loading DistilGPT-2 base model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    print("✅ Base model loaded!")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    dataset = RestaurantChatDataset(DATASET_FILE, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    
    train_model(model, train_loader, tokenizer, epochs=EPOCHS)
    
    save_model(model, tokenizer, OUTPUT_DIR)
    
    test_model(model, tokenizer)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"📁 Model Location: {OUTPUT_DIR}")
    print(f"📊 Training Examples: {len(dataset)}")
    print(f"🎯 Epochs Completed: {EPOCHS}")
    print("\n💡 Next Steps:")
    print("   1. Integrate this model into your main application")
    print("   2. Test with real customer reviews")
    print("   3. Fine-tune if needed with more specific data")
    print("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()