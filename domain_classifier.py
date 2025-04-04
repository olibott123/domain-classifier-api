import os
import re
import logging
import pickle
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    HAS_STOPWORDS = True
except:
    STOPWORDS = set()
    HAS_STOPWORDS = False

class DomainClassifier:
    def __init__(self, model_path=None, use_pinecone=False, pinecone_api_key=None, pinecone_index_name=None, confidence_threshold=0.6):
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.classes = None
        self.confidence_threshold = confidence_threshold
        self.use_pinecone = use_pinecone
        self.pinecone_index = None

        if use_pinecone:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = pc.Index(pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {pinecone_index_name}")

        if model_path:
            self.load_model(model_path)

    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = text.split()
        if HAS_STOPWORDS:
            tokens = [token for token in tokens if token not in STOPWORDS]
        return ' '.join(tokens)

    def classify_domain(self, domain_content, domain=None):
        if not self.classifier or not self.vectorizer:
            raise ValueError("Classifier not trained or loaded.")

        processed_text = self.preprocess_text(domain_content)

        if len(processed_text.split()) < 20:
            logger.warning(f"[{domain}] Content too short, skipping.")
            return {
                "predicted_class": "Unknown",
                "confidence_scores": {},
                "low_confidence": True,
                "detection_method": "content_length_check"
            }

        embedding = None
        if domain and self.use_pinecone:
            embedding = self.get_embedding(domain)

        if embedding is None:
            X = self.vectorizer.transform([processed_text])
            if domain and self.use_pinecone:
                embedding = X.toarray()[0]
                self.store_embedding(domain, embedding, metadata={
                    "domain": domain,
                    "content_length": len(domain_content),
                    "classification_date": datetime.now().isoformat()
                })
        else:
            X = embedding.reshape(1, -1)

        probabilities = self.classifier.predict_proba(X)[0]
        confidence_scores = {cls: float(prob) for cls, prob in zip(self.classes, probabilities)}
        predicted_class = self.classes[np.argmax(probabilities)]
        max_confidence = max(probabilities)
        low_confidence = max_confidence < self.confidence_threshold

        logger.info(f"Classified {domain}: {predicted_class} ({max_confidence:.2f})")

        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "max_confidence": float(max_confidence),
            "low_confidence": low_confidence,
            "detection_method": "model_classification"
        }

    def get_embedding(self, domain):
        if not self.use_pinecone or not self.pinecone_index:
            return None
        try:
            result = self.pinecone_index.fetch(ids=[domain])
            if domain in result.vectors:
                return np.array(result.vectors[domain].values)
            return None
        except:
            return None

    def store_embedding(self, domain, embedding, metadata=None):
        if not self.use_pinecone or not self.pinecone_index:
            return False
        metadata = metadata or {}
        try:
            self.pinecone_index.upsert([(domain, embedding.tolist(), metadata)])
            return True
        except:
            return False

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.classes = data['classes']
        self.confidence_threshold = data.get('confidence_threshold', 0.6)
        logger.info(f"Model loaded: {path}")

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'classes': self.classes,
                'confidence_threshold': self.confidence_threshold
            }, f)
        logger.info(f"Model saved: {path}")
