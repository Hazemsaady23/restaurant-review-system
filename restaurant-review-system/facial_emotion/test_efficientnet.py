#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = r"C:\Users\LENOVO\Desktop\Complete_Model\cnn_model\rafdb_efficientnetv2s_best.pth"
IMG_SIZE = 224
EMOTION_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path, num_classes=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = timm.create_model(
        "tf_efficientnetv2_s_in21k",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.4,
        drop_path_rate=0.2,
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from: {model_path}\n")
    return model, device


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

@torch.no_grad()
def predict_single_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)
    
    confidence, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), confidence.item(), probabilities[0].cpu().numpy()


def predict_image_file(model, image_path, transform, device, labels):
    try:
        image_tensor, original_image = preprocess_image(image_path, transform)
        predicted_class, confidence, probabilities = predict_single_image(model, image_tensor, device)
        
        emotion = labels[predicted_class]
        
        print(f"\nImage: {image_path}")
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("\nAll probabilities:")
        for i, (label, prob) in enumerate(zip(labels, probabilities)):
            print(f"  {label:12s}: {prob:.4f} ({prob*100:.2f}%)")
        
        return emotion, confidence, probabilities, original_image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_prediction(image, emotion, confidence, probabilities, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {emotion}\nConfidence: {confidence:.2%}', fontsize=12, fontweight='bold')
    
    colors = ['#FF6B6B' if i == np.argmax(probabilities) else '#4ECDC4' for i in range(len(labels))]
    bars = ax2.barh(labels, probabilities, color=colors)
    ax2.set_xlabel('Probability', fontsize=10)
    ax2.set_title('Emotion Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2%}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_directory(model, directory_path, transform, device, labels):
    print("\n" + "=" * 80)
    print(f"PROCESSING DIRECTORY: {directory_path}")
    print("=" * 80)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in os.listdir(directory_path) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print("No image files found in directory.")
        return
    
    results = []
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(directory_path, filename)
        print(f"\n[{i}/{len(image_files)}] {filename}")
        
        emotion, confidence, probabilities, _ = predict_image_file(
            model, image_path, transform, device, labels
        )
        
        if emotion:
            results.append({
                'filename': filename,
                'emotion': emotion,
                'confidence': confidence
            })
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        print(f"{result['filename']:30s} → {result['emotion']:12s} ({result['confidence']:.2%})")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(model, transform, device, labels):
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter image path to classify (or 'quit' to exit)\n")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not image_path:
            print("Please enter an image path.\n")
            continue
        
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}\n")
            continue
        
        emotion, confidence, probabilities, original_image = predict_image_file(
            model, image_path, transform, device, labels
        )
        
        if emotion:
            visualize = input("\nVisualize result? (y/n): ").strip().lower()
            if visualize == 'y':
                visualize_prediction(original_image, emotion, confidence, probabilities, labels)
        
        print()


# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

def detailed_analysis(model, image_path, transform, device, labels):
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    emotion, confidence, probabilities, original_image = predict_image_file(
        model, image_path, transform, device, labels
    )
    
    if emotion:
        sorted_indices = np.argsort(probabilities)[::-1]
        
        print("\n" + "-" * 80)
        print("Top 3 Predictions:")
        print("-" * 80)
        for rank, idx in enumerate(sorted_indices[:3], 1):
            print(f"{rank}. {labels[idx]:12s}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")
        
        print("\n" + "-" * 80)
        print("Statistics:")
        print("-" * 80)
        print(f"Mean probability: {np.mean(probabilities):.4f}")
        print(f"Std deviation:    {np.std(probabilities):.4f}")
        print(f"Max probability:  {np.max(probabilities):.4f}")
        print(f"Min probability:  {np.min(probabilities):.4f}")
        
        visualize_prediction(original_image, emotion, confidence, probabilities, labels)


# ============================================================================
# WEBCAM MODE (OPTIONAL)
# ============================================================================

def webcam_mode(model, transform, device, labels):
    print("\n" + "=" * 80)
    print("WEBCAM MODE")
    print("=" * 80)
    
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) not installed. Install with: pip install opencv-python")
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 's' to save prediction")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        image_tensor = transform(pil_image).unsqueeze(0)
        
        predicted_class, confidence, probabilities = predict_single_image(model, image_tensor, device)
        emotion = labels[predicted_class]
        
        cv2.putText(frame, f"{emotion}: {confidence:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{emotion}_{confidence:.2f}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    model, device = load_model(MODEL_PATH)
    transform = get_transform()
    
    while True:
        print("\n" + "=" * 80)
        print("EMOTION RECOGNITION TESTING MENU")
        print("=" * 80)
        print("1. Predict single image")
        print("2. Process directory (batch)")
        print("3. Interactive mode")
        print("4. Detailed analysis")
        print("5. Webcam mode (real-time)")
        print("6. Exit")
        print("=" * 80)
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                emotion, confidence, probabilities, original_image = predict_image_file(
                    model, image_path, transform, device, EMOTION_LABELS
                )
                if emotion:
                    visualize = input("\nVisualize result? (y/n): ").strip().lower()
                    if visualize == 'y':
                        visualize_prediction(original_image, emotion, confidence, probabilities, EMOTION_LABELS)
            else:
                print(f"Error: File not found: {image_path}")
        
        elif choice == '2':
            directory_path = input("Enter directory path: ").strip()
            if os.path.isdir(directory_path):
                process_directory(model, directory_path, transform, device, EMOTION_LABELS)
            else:
                print(f"Error: Directory not found: {directory_path}")
        
        elif choice == '3':
            interactive_mode(model, transform, device, EMOTION_LABELS)
        
        elif choice == '4':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                detailed_analysis(model, image_path, transform, device, EMOTION_LABELS)
            else:
                print(f"Error: File not found: {image_path}")
        
        elif choice == '5':
            webcam_mode(model, transform, device, EMOTION_LABELS)
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-6.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--image" and len(sys.argv) > 2:
            model, device = load_model(MODEL_PATH)
            transform = get_transform()
            predict_image_file(model, sys.argv[2], transform, device, EMOTION_LABELS)
        elif sys.argv[1] == "--dir" and len(sys.argv) > 2:
            model, device = load_model(MODEL_PATH)
            transform = get_transform()
            process_directory(model, sys.argv[2], transform, device, EMOTION_LABELS)
        elif sys.argv[1] == "--webcam":
            model, device = load_model(MODEL_PATH)
            transform = get_transform()
            webcam_mode(model, transform, device, EMOTION_LABELS)
        else:
            print("Usage:")
            print("  python test_model.py                    # Show menu")
            print("  python test_model.py --image <path>     # Test single image")
            print("  python test_model.py --dir <path>       # Test directory")
            print("  python test_model.py --webcam           # Real-time webcam")
    else:
        main()
