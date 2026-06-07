# restaurant-review-system

A comprehensive multi-modal AI system for analyzing restaurant reviews through facial emotion detection, sentiment analysis, and interactive chatbot capabilities.

## 🎯 Project Overview

This system combines three deep learning models to provide a complete restaurant review analysis solution:

1. **Facial Emotion Detection** - ResNet-based model using EfficientNetV2-S on RAF-DB dataset
2. **Sentiment Analysis** - DistilBERT model for text sentiment classification
3. **Interactive Chatbot** - DistilGPT-2 powered conversational agent

## 🏗️ Project Structure

```
restaurant-review-system/
├── README.md
├── .gitignore
├── streamlit.py                          # Main Streamlit application
├── facial_emotion/
│   ├── Raf_Db_Model.py                   # Facial emotion detection model
│   ├── Raf_db_analysis.py                # Analysis and evaluation
│   ├── test_efficientnet.py              # Testing script
│   ├── requirements_efficientnet.txt     # Dependencies
│   └── rafdb_efficientnetv2s_best.pth    # Model weights (not in repo)
├── sentiment_analysis/
│   ├── dist.py                           # DistilBERT sentiment model
│   ├── Dist_analysis.py                  # Analysis and evaluation
│   ├── test_model_Dist.py                # Testing script
│   ├── requirements_distilbert.txt       # Dependencies
│   └── distilbert_model.safetensors      # Model weights (not in repo)
├── chatbot/
│   ├── Untitled-1.py                     # DistilGPT-2 chatbot
│   ├── Chatbot_analysis.py               # Analysis and evaluation
│   ├── test_chatbot.py                   # Testing script
│   ├── requirements_chatbot.txt          # Dependencies
│   └── distilgpt2_model.safetensors      # Model weights (not in repo)
└── datasets/                             # Dataset files (not in repo)
```

## 🚀 Features

### 1. Facial Emotion Detection
- Built on EfficientNetV2-S architecture
- Trained on RAF-DB (Real-world Affective Faces Database)
- Detects emotions from customer facial expressions
- Real-time emotion classification

### 2. Sentiment Analysis
- DistilBERT-based text classification
- Analyzes written reviews for sentiment
- Fast and accurate sentiment predictions
- Supports multiple review formats

### 3. Conversational Chatbot
- Powered by DistilGPT-2
- Interactive customer engagement
- Contextual response generation
- Natural language understanding

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for inference)
- 8GB+ RAM

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Hazemsaady23/restaurant-review-system.git
cd restaurant-review-system
```

### Step 2: Download Model Files

**⚠️ Important:** Model files are not included in this repository due to size constraints (300MB+).

**Download the complete project with models from:**
- 🔗 [Google Drive Link](https://drive.google.com/drive/folders/1K29ZaIZp1lAQjtG3ZFgXklMbU5vg3ptc?usp=drive_link)

After downloading, place the model files in their respective directories:
- `rafdb_efficientnetv2s_best.pth` → `facial_emotion/`
- `distilbert_model.safetensors` → `sentiment_analysis/`
- `distilgpt2_model.safetensors` → `chatbot/`

### Step 3: Install Dependencies

You can install dependencies for all models or individually:

**Install all dependencies:**
```bash
pip install -r facial_emotion/requirements_efficientnet.txt
pip install -r sentiment_analysis/requirements_distilbert.txt
pip install -r chatbot/requirements_chatbot.txt
```

**Or install individually per model as needed.**

### Step 4: Run the Application

```bash
streamlit run streamlit.py
```

## 🧪 Testing

Each model includes its own test script:

```bash
# Test facial emotion detection
python facial_emotion/test_efficientnet.py

# Test sentiment analysis
python sentiment_analysis/test_model_Dist.py

# Test chatbot
python chatbot/test_chatbot.py
```

## 📊 Model Performance

### Facial Emotion Detection (EfficientNetV2-S)
- Architecture: EfficientNetV2-S with custom classification head
- Dataset: RAF-DB
- Performance metrics available in `Raf_db_analysis.py`

### Sentiment Analysis (DistilBERT)
- Architecture: DistilBERT
- Fine-tuned for sentiment classification
- Performance metrics available in `Dist_analysis.py`

### Chatbot (DistilGPT-2)
- Architecture: DistilGPT-2
- Conversational AI for customer interaction
- Performance metrics available in `Chatbot_analysis.py`

## 💻 Usage Examples

### Using Individual Models


```

### Using Streamlit Interface

Simply run `streamlit run streamlit.py` and interact with all three models through the web interface.

## 📁 Dataset Information

The project uses the following datasets:
- **RAF-DB**: Real-world Affective Faces Database for emotion detection
- **Custom review dataset**: For sentiment analysis training
- Dataset files are not included in the repository

## 🛠️ Technologies Used

- **Deep Learning Frameworks**: PyTorch, Transformers
- **Models**: EfficientNetV2-S, DistilBERT, DistilGPT-2
- **Frontend**: Streamlit
- **Languages**: Python 3.8+

## 📝 Project Components

### Analysis Scripts
Each model includes an analysis script that provides:
- Model evaluation metrics
- Performance visualization
- Detailed analysis results

### Test Scripts
Comprehensive testing scripts for:
- Model inference validation
- Performance benchmarking
- Error analysis

## 🤝 Contributing

This is a portfolio/interview project. However, suggestions and feedback are welcome!

## 📄 License

This project is created for educational and interview purposes.

## 👤 Author

**Hazem Mahmoud**
- GitHub: (https://github.com/Hazemsaady23)
- LinkedIn: (https://linkedin.com/in/hazem-saad97)
- Email: Hazemsaady18@gmail.com

## 🙏 Acknowledgments

- RAF-DB dataset creators
- Hugging Face for pre-trained models
- PyTorch and Transformers library maintainers

## 📞 Contact

For any questions or discussions about this project, please reach out via email or LinkedIn.

---

