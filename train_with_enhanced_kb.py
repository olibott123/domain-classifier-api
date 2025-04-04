#!/usr/bin/env python3
"""
Script to train an improved domain classifier model using the enhanced knowledge base.
This model will have better accuracy for classifying domains, especially MSPs.
"""
import os
import pandas as pd
import numpy as np
import logging
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_enhanced_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to enhanced knowledge base
ENHANCED_KB_PATH = "knowledge_base_clean.csv"
OUTPUT_MODEL_PATH = "domain_classifier_model_enhanced.pkl"

def preprocess_text(text):
    """
    Enhanced text preprocessing with domain-specific handling.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Handle domain-specific terms (preserve them)
    # For IT/MSP terms
    text = text.replace("it department", "it_department")
    text = text.replace("help desk", "help_desk")
    text = text.replace("managed service", "managed_service")
    text = text.replace("cloud solution", "cloud_solution")
    text = text.replace("managed service provider", "managed_service_provider")
    text = text.replace("msp", "managed_service_provider")
    text = text.replace("it support", "it_support")
    text = text.replace("network monitoring", "network_monitoring")
    
    # For A/V terms
    text = text.replace("home theater", "home_theater")
    text = text.replace("smart home", "smart_home")
    text = text.replace("audio video", "audio_video")
    text = text.replace("av", "audio_video")
    text = text.replace("a/v", "audio_video")
    text = text.replace("commercial audio", "commercial_audio")
    text = text.replace("digital signage", "digital_signage")
    text = text.replace("residential", "residential_av")
    text = text.replace("commercial", "commercial_av")
    
    # Remove special characters but keep the compound terms
    text = re.sub(r'[^\w\s_]', '', text)
    
    # Remove standalone digits (but keep digits in words)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Split and join to normalize spacing
    tokens = text.split()
    
    # Remove very short tokens (likely not informative)
    tokens = [token for token in tokens if len(token) > 1]
    
    return ' '.join(tokens)

def train_enhanced_model():
    """Train an enhanced model with the enriched knowledge base."""
    logger.info("Starting enhanced model training")
    
    # Load enhanced knowledge base
    try:
        logger.info(f"Loading enhanced knowledge base from: {ENHANCED_KB_PATH}")
        kb_df = pd.read_csv(ENHANCED_KB_PATH)
        logger.info(f"Loaded {len(kb_df)} domains")
        
        # Check columns
        logger.info(f"Columns: {kb_df.columns.tolist()}")
        
        # Verify required columns exist
        if 'content' not in kb_df.columns:
            logger.error("Required column 'content' not found in knowledge base")
            return
        
        # Use either 'company_type' or 'category' column
        category_col = 'company_type' if 'company_type' in kb_df.columns else 'category'
        if category_col not in kb_df.columns:
            logger.error(f"Required column '{category_col}' not found in knowledge base")
            return
        
        # Show category distribution
        logger.info("Category distribution:")
        category_counts = kb_df[category_col].value_counts()
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} examples")
        
        # Preprocess text
        logger.info("Preprocessing text with enhanced preprocessing...")
        kb_df['processed_content'] = kb_df['content'].apply(preprocess_text)
        
        # Create TF-IDF vectorizer with improved parameters
        logger.info("Creating TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=3000,      # More features
            min_df=2,               # Ignore very rare terms
            max_df=0.9,             # Ignore very common terms
            ngram_range=(1, 2),     # Include bigrams
            sublinear_tf=True       # Apply sublinear TF scaling
        )
        X = vectorizer.fit_transform(kb_df['processed_content'])
        
        # Encode labels
        logger.info("Encoding labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(kb_df[category_col])
        classes = label_encoder.classes_
        logger.info(f"Classes: {classes}")
        
        # Calculate class weights (inverse of frequency)
        class_counts = np.bincount(y)
        total_samples = len(y)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        logger.info(f"Class weights: {class_weights}")
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create Random Forest classifier with optimized parameters
        logger.info("Training Random Forest classifier...")
        rf = RandomForestClassifier(
            n_estimators=300,       # More trees for better accuracy
            max_depth=25,           # Deeper trees to capture more patterns
            class_weight=class_weights,
            random_state=42,
            min_samples_leaf=1,
            min_samples_split=2,
            bootstrap=True,
            max_features='sqrt'
        )
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_weighted')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f}")
        
        # Train on full training set
        rf.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = rf.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        
        logger.info(f"Test set accuracy: {report['accuracy']:.4f}")
        logger.info("Classification report:")
        for category in classes:
            metrics = report[category]
            logger.info(f"  {category}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1-score: {metrics['f1-score']:.4f}")
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion matrix:")
        for i, row in enumerate(cm):
            logger.info(f"  {classes[i]}: {row}")
        
        # Save model
        logger.info(f"Saving model to {OUTPUT_MODEL_PATH}")
        model_data = {
            'vectorizer': vectorizer,
            'classifier': rf,
            'label_encoder': label_encoder,
            'classes': classes,
            'confidence_threshold': 0.6
        }
        
        with open(OUTPUT_MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully!")
        
        # Test on example texts
        logger.info("Testing model on example texts...")
        
        test_examples = [
            # MSP example
            ("MSP Example", """We are a trusted managed IT service provider serving small and medium businesses.
            Our services include help desk support, network monitoring, cloud solutions, cybersecurity,
            and disaster recovery planning. Our team of certified technicians provides 24/7 monitoring
            and responsive support to keep your business running smoothly."""),
            
            # IT Department example
            ("IT Department Example", """Welcome to the company IT portal. Employees can request technical support,
            access corporate applications, reset passwords, and view IT policies. Our internal IT team
            supports all corporate technology needs including workstation setup, network access,
            email configuration, and software deployment."""),
            
            # Residential A/V example
            ("Residential A/V Example", """Our home theater installation team creates immersive entertainment experiences
            for discerning homeowners. We specialize in luxury home automation, multi-room audio systems,
            custom theater rooms, and smart home integration. Our certified designers will transform your
            living space with the latest in audio and video technology."""),
            
            # Commercial A/V example
            ("Commercial A/V Example", """We provide commercial audio-visual solutions for businesses, education,
            and government facilities. Our services include conference room design, digital signage,
            video walls, sound reinforcement, and control system programming. We help organizations
            communicate effectively through integrated technology systems.""")
        ]
        
        for name, text in test_examples:
            # Preprocess
            processed = preprocess_text(text)
            X_example = vectorizer.transform([processed])
            
            # Predict
            probs = rf.predict_proba(X_example)[0]
            predicted_class_idx = np.argmax(probs)
            predicted_class = classes[predicted_class_idx]
            confidence = probs[predicted_class_idx]
            
            logger.info(f"\n{name}:")
            logger.info(f"  Classified as: {predicted_class}")
            logger.info(f"  Confidence: {confidence:.4f}")
            
            # All probabilities
            logger.info("  All probabilities:")
            for i, cls in enumerate(classes):
                logger.info(f"    {cls}: {probs[i]:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Make sure enhanced KB exists
    if not os.path.exists(ENHANCED_KB_PATH):
        logger.error(f"Enhanced knowledge base not found at: {ENHANCED_KB_PATH}")
        logger.error("Please run crawl_only_knowledge_base.py first to create it")
        exit(1)
    
    success = train_enhanced_model()
    
    if success:
        logger.info(f"Training completed successfully! New model saved to: {OUTPUT_MODEL_PATH}")
        logger.info("To use this model, update your API_SERVICE.py MODEL_PATH to point to it")
    else:
        logger.error("Training failed. Check the logs for details.")
