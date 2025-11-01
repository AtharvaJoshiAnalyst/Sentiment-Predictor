from flask import Flask, request, jsonify, render_template
from textblob import TextBlob as TB
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# ------------------ TextBlob Setup ------------------
def get_textblob_sentiment(text):
    polarity = TB(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# ------------------ VADER Setup ------------------
vader_analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "Neutral"
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.08:
            return 'Positive'
        elif compound <= -0.08:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        print(f"VADER Error: {e}")
        return "Neutral"

# ------------------ Transformer (BERT) Setup ------------------
# Use a smaller, faster model for deployment
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Cache directory for models
cache_dir = os.path.join(os.getcwd(), 'model_cache')
os.makedirs(cache_dir, exist_ok=True)

try:
    # Load model with proper device handling
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    
    bert_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=device
    )
    print("✓ BERT model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load BERT model: {e}")
    bert_pipeline = None

def get_bert_sentiment(text):
    try:
        if bert_pipeline is None:
            return "Unavailable"
        
        if not isinstance(text, str) or not text.strip():
            return "Neutral"
        
        result = bert_pipeline(text[:512])[0]
        label = result['label'].upper()
        
        if 'POSITIVE' in label:
            return 'Positive'
        elif 'NEGATIVE' in label:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        print(f"BERT Error: {e}")
        return "Neutral"

# ------------------ Routes ------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        text = request.form.get('text') or (request.json and request.json.get('text'))
        if not text:
            return jsonify({'error': 'No text provided!'}), 400
        
        return jsonify({
            'text': text,
            'textblob_sentiment': get_textblob_sentiment(text),
            'vader_sentiment': get_vader_sentiment(text),
            'bert_sentiment': get_bert_sentiment(text)
        })
    else:
        return jsonify({
            'message': 'Send a POST request with text input to get sentiment prediction.',
            'example': {'text': 'I love Python!'}
        })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)