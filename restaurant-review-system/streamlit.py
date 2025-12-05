"""
Restaurant Review AI System
Professional Interface with Emotion Detection, Sentiment Analysis & AI Response
"""

import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import warnings
import os
import timm

# Transformers import with fallback
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        GPT2Tokenizer,
        GPT2LMHeadModel
    )
except ImportError:
    st.error("❌ Missing transformers library. Run: pip install transformers")
    st.stop()

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\LENOVO\Desktop\Complete_Model"
RAFDB_MODEL_PATH = r"C:\Users\LENOVO\Desktop\RAF-DB\rafdb_efficientnetv2s_best.pth"  # Updated path
DISTILBERT_PATH = os.path.join(BASE_DIR, "DistilBert")
CHATBOT_PATH = os.path.join(BASE_DIR, "restaurant_chatbot_model")

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': '#ef4444', 'Disgust': '#8b5cf6', 'Fear': '#f59e0b',
    'Happy': '#10b981', 'Neutral': '#6b7280', 'Sad': '#3b82f6', 'Surprise': '#ec4899'
}
EMOTION_EMOJIS = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
}

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Restaurant Review AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PROFESSIONAL DARK THEME (Same as before)
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    }
    h1 {
        color: #f59e0b !important;
        font-weight: 700 !important;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
    }
    h2, h3 {
        color: #fbbf24 !important;
        font-weight: 600 !important;
    }
    p, span, label, div {
        color: #e2e8f0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(245, 158, 11, 0.5);
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    }
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 2px solid #334155 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2) !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 2px dashed #475569;
        border-radius: 10px;
        padding: 1.5rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #f59e0b;
    }
    .stSuccess {
        background-color: #064e3b !important;
        border-left: 4px solid #10b981 !important;
        color: #d1fae5 !important;
    }
    .stInfo {
        background-color: #1e3a8a !important;
        border-left: 4px solid #3b82f6 !important;
        color: #dbeafe !important;
    }
    .stError {
        background-color: #7f1d1d !important;
        border-left: 4px solid #ef4444 !important;
        color: #fecaca !important;
    }
    .stWarning {
        background-color: #78350f !important;
        border-left: 4px solid #f59e0b !important;
        color: #fde68a !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white !important;
    }
    [data-testid="column"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #334155;
    }
    img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stSpinner > div {
        border-color: #f59e0b transparent transparent transparent !important;
    }
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    .stSlider > div > div > div {
        background-color: #f59e0b !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def preprocess_face_rafdb(face_img):
    """Preprocess face for RAF-DB EfficientNetV2 model (224x224, RGB, normalized)"""
    # Convert to RGB if grayscale
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
    
    # Resize to 224x224
    face_img = cv2.resize(face_img, (224, 224))
    
    # Convert to tensor and normalize (ImageNet stats)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = (face_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # HWC to CHW
    face_img = np.transpose(face_img, (2, 0, 1))
    
    return torch.from_numpy(face_img).unsqueeze(0).float()

def detect_emotion(model, face_img, device):
    """Detect emotion from face image using RAF-DB model"""
    with torch.no_grad():
        face_tensor = preprocess_face_rafdb(face_img).to(device)
        outputs = model(face_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return EMOTIONS[pred.item()], conf.item() * 100, probs[0].cpu().numpy()

def analyze_sentiment(tokenizer, model, text, device):
    """Analyze sentiment of text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, 1)
    return ("Positive" if pred.item() == 1 else "Negative"), conf.item() * 100

def generate_response(tokenizer, model, text, device, max_len=120, temp=0.8, top_p=0.92):
    """Generate chatbot response"""
    prompt = f"Customer: {text} <|endoftext|> Bot:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=torch.ones_like(inputs['input_ids']),
            max_length=max_len,
            min_length=20,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=top_p,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()
    else:
        response = response.replace(prompt, "").strip()
    
    return ' '.join(response.split())

def detect_faces(image):
    """Detect faces in image"""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    return faces, img_array

# ==========================================
# MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models(device):
    """Load all AI models"""
    # Load RAF-DB EfficientNetV2
    rafdb_model = timm.create_model(
        "tf_efficientnetv2_s_in21k",
        pretrained=False,
        num_classes=7,
        drop_rate=0.4,
        drop_path_rate=0.2,
    )
    checkpoint = torch.load(RAFDB_MODEL_PATH, map_location=device, weights_only=True)
    rafdb_model.load_state_dict(checkpoint)
    rafdb_model.to(device).eval()
    
    # Load DistilBERT
    sent_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
    sent_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)
    sent_model.to(device).eval()
    
    # Load Chatbot
    chat_tokenizer = GPT2Tokenizer.from_pretrained(CHATBOT_PATH, local_files_only=True)
    chat_model = GPT2LMHeadModel.from_pretrained(CHATBOT_PATH, local_files_only=True)
    chat_tokenizer.pad_token = chat_tokenizer.eos_token
    chat_model.to(device).eval()
    
    return rafdb_model, sent_tokenizer, sent_model, chat_tokenizer, chat_model

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # Title
    st.markdown("# 🍽️ Restaurant Review AI System")
    st.markdown("### Advanced Customer Feedback Analysis with AI")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ System Status")
        
        # Check paths
        rafdb_ok = os.path.exists(RAFDB_MODEL_PATH)
        dist_ok = os.path.exists(DISTILBERT_PATH)
        chat_ok = os.path.exists(CHATBOT_PATH)
        
        st.markdown("### 📦 Models")
        st.write("✅ RAF-DB EfficientNetV2" if rafdb_ok else "❌ RAF-DB EfficientNetV2")
        st.write("✅ DistilBERT" if dist_ok else "❌ DistilBERT")
        st.write("✅ Chatbot GPT-2" if chat_ok else "❌ Chatbot GPT-2")
        
        if not all([rafdb_ok, dist_ok, chat_ok]):
            st.error("⚠️ Some models are missing!")
            with st.expander("Show Details"):
                if not rafdb_ok:
                    st.text(f"RAF-DB: {RAFDB_MODEL_PATH}")
                if not dist_ok:
                    st.text(f"DistilBERT: {DISTILBERT_PATH}")
                if not chat_ok:
                    st.text(f"Chatbot: {CHATBOT_PATH}")
            st.stop()
        
        st.markdown("---")
        
        # Device info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_emoji = "🚀" if device.type == 'cuda' else "💻"
        st.info(f"{device_emoji} **Device:** {device.type.upper()}")
        
        st.markdown("---")
        
        # Chatbot settings
        st.markdown("### 🤖 Chatbot Settings")
        max_length = st.slider("Response Length", 60, 200, 120, 10)
        temperature = st.slider("Creativity", 0.2, 1.2, 0.8, 0.1)
        top_p = st.slider("Diversity", 0.5, 1.0, 0.92, 0.02)
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("""
        - 😊 **Emotion Detection** (RAF-DB EfficientNetV2)
        - 💬 **Sentiment Analysis** (DistilBERT)
        - 🤖 **AI Response** (GPT-2)
        - 📸 **Webcam & Upload** support
        """)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with st.spinner('🔄 Loading AI models...'):
        try:
            rafdb_model, sent_tok, sent_model, chat_tok, chat_model = load_models(device)
            st.success("✅ All models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📸 Face + Review Analysis", "💬 Text Only Analysis", "📊 Batch Analysis"])
    
    # ==========================================
    # TAB 1: FACE + REVIEW
    # ==========================================
    with tab1:
        st.markdown("## Complete Customer Analysis")
        st.markdown("Analyze both facial emotion and written review together")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Step 1: Capture Face")
            
            input_method = st.radio(
                "Choose input method:",
                ["📤 Upload Image", "📷 Use Webcam"],
                horizontal=True
            )
            
            image = None
            
            if input_method == "📤 Upload Image":
                uploaded = st.file_uploader(
                    "Upload a face image",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear front-facing photo"
                )
                if uploaded:
                    image = Image.open(uploaded)
            else:
                camera = st.camera_input("Take a photo")
                if camera:
                    image = Image.open(camera)
            
            emotion = None
            emotion_conf = 0
            all_probs = None
            
            if image:
                st.image(image, caption="Input Image", use_container_width=True)
                
                with st.spinner("🔍 Analyzing facial emotion..."):
                    faces, img_array = detect_faces(image)
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = img_array[y:y+h, x:x+w]
                        emotion, emotion_conf, all_probs = detect_emotion(rafdb_model, face_roi, device)
                        
                        # Display emotion result
                        emoji = EMOTION_EMOJIS[emotion]
                        color = EMOTION_COLORS[emotion]
                        
                        st.markdown(f"""
                        <div style="background: {color}22; border-left: 4px solid {color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <h3 style="color: {color} !important; margin: 0;">{emoji} {emotion}</h3>
                            <p style="color: #e2e8f0 !important; margin: 0.5rem 0 0 0;">Confidence: {emotion_conf:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show all probabilities
                        with st.expander("📊 View All Emotion Probabilities"):
                            for i, em in enumerate(EMOTIONS):
                                prob = all_probs[i] * 100
                                st.progress(prob / 100, text=f"{EMOTION_EMOJIS[em]} {em}: {prob:.1f}%")
                    else:
                        st.warning("⚠️ No face detected. Please try a clearer image.")
        
        with col2:
            st.markdown("### ✍️ Step 2: Enter Review")
            
            review_text = st.text_area(
                "Customer review:",
                height=200,
                placeholder="Example: The food was amazing but the service was slow..."
            )
            
            if st.button("🚀 Analyze Complete Review", type="primary", use_container_width=True):
                if not review_text:
                    st.error("⚠️ Please enter a review!")
                elif not emotion:
                    st.error("⚠️ Please capture a face image first!")
                else:
                    with st.spinner("🔍 Analyzing sentiment and generating response..."):
                        sentiment, sent_conf = analyze_sentiment(sent_tok, sent_model, review_text, device)
                        bot_response = generate_response(
                            chat_tok, chat_model, review_text, device,
                            max_len=max_length, temp=temperature, top_p=top_p
                        )
                    
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Metrics
                    metric1, metric2, metric3 = st.columns(3)
                    with metric1:
                        st.metric("Emotion", f"{EMOTION_EMOJIS[emotion]} {emotion}", f"{emotion_conf:.1f}% confidence")
                    with metric2:
                        st.metric("Sentiment", sentiment, f"{sent_conf:.1f}% confidence")
                    with metric3:
                        avg_conf = (emotion_conf + sent_conf) / 2
                        st.metric("Overall Confidence", f"{avg_conf:.1f}%")
                    
                    # AI Response
                    st.markdown("### 🤖 AI-Generated Response")
                    st.info(bot_response)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommended Actions")
                    if sentiment == "Positive" and emotion in ["Happy", "Surprise"]:
                        st.success("✅ **Great experience!** Thank customer and encourage return visit.")
                    elif sentiment == "Negative" and emotion in ["Angry", "Disgust"]:
                        st.error("⚠️ **Critical issue!** Respond immediately with apology and solution.")
                    else:
                        st.warning("📝 **Mixed signals.** Respond professionally and ask for clarification.")
    
    # ==========================================
    # TAB 2: TEXT ONLY
    # ==========================================
    with tab2:
        st.markdown("## Text-Based Sentiment Analysis")
        st.markdown("Analyze customer reviews without facial emotion")
        
        text_review = st.text_area(
            "Enter review text:",
            height=250,
            placeholder="Type or paste customer review here..."
        )
        
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if not text_review:
                st.error("⚠️ Please enter a review!")
            else:
                with st.spinner("🔍 Analyzing..."):
                    sentiment, conf = analyze_sentiment(sent_tok, sent_model, text_review, device)
                    response = generate_response(
                        chat_tok, chat_model, text_review, device,
                        max_len=max_length, temp=temperature, top_p=top_p
                    )
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Sentiment Analysis")
                    if sentiment == "Positive":
                        st.success(f"✅ **{sentiment}**")
                        st.progress(conf / 100)
                        st.caption(f"Confidence: {conf:.1f}%")
                        st.markdown("""
                        **Insights:**
                        - Customer is satisfied
                        - Good opportunity for loyalty
                        - Consider thank you message
                        """)
                    else:
                        st.error(f"⚠️ **{sentiment}**")
                        st.progress(conf / 100)
                        st.caption(f"Confidence: {conf:.1f}%")
                        st.markdown("""
                        **Insights:**
                        - Customer is dissatisfied
                        - Requires immediate attention
                        - Offer apology and solution
                        """)
                
                with col2:
                    st.markdown("### 🤖 AI Response")
                    st.info(response)
                    
                    if st.button("📋 Copy Response"):
                        st.toast("✅ Response copied to clipboard!")
    
    # ==========================================
    # TAB 3: BATCH ANALYSIS
    # ==========================================
    with tab3:
        st.markdown("## Batch Review Analysis")
        st.markdown("Analyze multiple reviews at once")
        
        batch_input = st.text_area(
            "Enter multiple reviews (one per line):",
            height=300,
            placeholder="Review 1: Great food!\nReview 2: Service was slow...\nReview 3: Amazing experience!"
        )
        
        if st.button("🚀 Analyze All Reviews", type="primary", use_container_width=True):
            if not batch_input:
                st.error("⚠️ Please enter some reviews!")
            else:
                reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
                
                st.markdown(f"### Analyzing {len(reviews)} reviews...")
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, review in enumerate(reviews):
                    with st.spinner(f"Processing review {idx+1}/{len(reviews)}..."):
                        sent, conf = analyze_sentiment(sent_tok, sent_model, review, device)
                        results.append({
                            'Review': review[:50] + '...' if len(review) > 50 else review,
                            'Sentiment': sent,
                            'Confidence': f"{conf:.1f}%"
                        })
                    progress_bar.progress((idx + 1) / len(reviews))
                
                st.success(f"✅ Analyzed {len(reviews)} reviews!")
                
                # Summary
                positive = sum(1 for r in results if r['Sentiment'] == 'Positive')
                negative = len(results) - positive
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(reviews))
                with col2:
                    st.metric("Positive", positive, f"{positive/len(reviews)*100:.0f}%")
                with col3:
                    st.metric("Negative", negative, f"{negative/len(reviews)*100:.0f}%")
                
                # Results table
                st.markdown("### Detailed Results")
                st.dataframe(results, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("🍽️ Restaurant Review AI | Powered by RAF-DB EfficientNetV2, DistilBERT & GPT-2")

if __name__ == "__main__":
    main()