import os
import time
import pandas as pd
from datetime import datetime
import logging
from domain_classifier import DomainClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("domain_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration - using your existing Snowflake connection details
SNOWFLAKE_CONFIG = {
    'user': "url_domain_crawler_testing_user",
    'private_key_path': "/root/crawler_api/rsa_key.der",  # Match your current key format
    'account': "DOMOTZ-MAIN",
    'warehouse': "TESTING_WH",  # Using your testing warehouse
    'database': 'DOMOTZ_TESTING_SOURCE',
    'schema': 'EXTERNAL_PUSH'
}

# Match the table names to your existing structure
SOURCE_TABLE = 'DOMAIN_CONTENT'
TARGET_TABLE = 'DOMAIN_CLASSIFICATION'
MODEL_PATH = 'domain_classifier_model.pkl'
KNOWLEDGE_BASE_PATH = 'knowledge_base/knowledge_base.csv'

def train_classifier():
    """Train the classifier with the knowledge base"""
    logger.info("Training classifier with knowledge base...")
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        logger.error(f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}")
        logger.error("Please run knowledge_base_builder.py first")
        raise FileNotFoundError(f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}")
        
    classifier = DomainClassifier()
    classifier.train_with_knowledge_base(KNOWLEDGE_BASE_PATH)
    classifier.save_model(MODEL_PATH)
    logger.info(f"Classifier trained and saved to {MODEL_PATH}")
    
    return classifier

def run_classification_service(interval_hours=24):
    """Run the classification service at specified interval"""
    if not os.path.exists(MODEL_PATH):
        logger.info("No existing model found. Training new classifier...")
        classifier = train_classifier()
    else:
        logger.info(f"Loading existing classifier from {MODEL_PATH}")
        classifier = DomainClassifier(model_path=MODEL_PATH)
    
    while True:
        try:
            start_time = datetime.now()
            logger.info(f"Starting classification run at {start_time}")
            
            # Process domains from Snowflake
            classifier.batch_classify_from_snowflake(
                SNOWFLAKE_CONFIG,
                SOURCE_TABLE,
                TARGET_TABLE
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Classification run completed in {duration:.2f} seconds")
            
            # Wait for next interval
            logger.info(f"Waiting {interval_hours} hours until next run")
            time.sleep(interval_hours * 3600)
            
        except Exception as e:
            logger.error(f"Error in classification run: {e}")
            logger.info("Will retry in 1 hour")
            time.sleep(3600)

def run_single_classification():
    """Run a single classification pass and exit"""
    if not os.path.exists(MODEL_PATH):
        logger.info("No existing model found. Training new classifier...")
        classifier = train_classifier()
    else:
        logger.info(f"Loading existing classifier from {MODEL_PATH}")
        classifier = DomainClassifier(model_path=MODEL_PATH)
    
    try:
        start_time = datetime.now()
        logger.info(f"Starting classification run at {start_time}")
        
        # Process domains from Snowflake
        classifier.batch_classify_from_snowflake(
            SNOWFLAKE_CONFIG,
            SOURCE_TABLE,
            TARGET_TABLE
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Classification run completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in classification run: {e}")
        raise

if __name__ == "__main__":
    # You can change this to run_single_classification() if you want it to run once and exit
    run_classification_service()
