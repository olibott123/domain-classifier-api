import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import snowflake.connector
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union

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

# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

class DomainClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the domain classifier
        
        Args:
            model_path: Path to saved model, if None a new model will be trained
        """
        # Define company types
        self.company_types = [
            'Managed Service Provider',
            'Integrator - Residential A/V', 
            'Integrator - Commercial A/V',
            'IT Department/Enterprise'
        ]
        
        # Load embedding model
        try:
            logger.info("Loading sentence embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Set up text preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize classifier
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.classifier = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text from websites
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = [self.lemmatizer.lemmatize(word) for word in text.split() 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return np.zeros(384)  # Return zero vector if text is empty
        return self.embedding_model.encode(processed_text)
    
    def connect_to_snowflake(self, config: Dict[str, str]) -> snowflake.connector.SnowflakeConnection:
        """
        Connect to Snowflake database
        
        Args:
            config: Connection configuration
            
        Returns:
            Snowflake connection
        """
        logger.info(f"Connecting to Snowflake: {config['user']}@{config['account']}")
        
        try:
            if 'private_key_path' in config and os.path.exists(config['private_key_path']):
                # Use key pair authentication
                logger.info(f"Using private key at {config['private_key_path']}")
                
                # Load private key in binary mode
                with open(config['private_key_path'], 'rb') as key_file:
                    private_key = key_file.read()
                
                return snowflake.connector.connect(
                    user=config['user'],
                    private_key=private_key,
                    account=config['account'],
                    warehouse=config['warehouse'],
                    database=config['database'],
                    schema=config['schema'],
                    authenticator='snowflake_jwt'
                )
            elif 'password' in config:
                # Use password authentication
                logger.info("Using password authentication")
                return snowflake.connector.connect(
                    user=config['user'],
                    password=config['password'],
                    account=config['account'],
                    warehouse=config['warehouse'],
                    database=config['database'],
                    schema=config['schema']
                )
            else:
                raise ValueError("No valid authentication method provided in config")
        except Exception as e:
            logger.error(f"Snowflake connection error: {e}")
            raise
    
    def load_knowledge_base_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load knowledge base data from CSV
        
        Args:
            csv_path: Path to knowledge base CSV
            
        Returns:
            DataFrame with knowledge base data
        """
        logger.info(f"Loading knowledge base from {csv_path}")
        kb_data = pd.read_csv(csv_path)
        
        required_columns = ['domain', 'company_type', 'content']
        for col in required_columns:
            if col not in kb_data.columns:
                raise ValueError(f"Required column '{col}' not found in knowledge base CSV")
        
        return kb_data
    
    def load_knowledge_base_from_snowflake(self, 
                                          snowflake_config: Dict[str, str], 
                                          table_name: str) -> pd.DataFrame:
        """
        Load knowledge base data from Snowflake
        
        Args:
            snowflake_config: Snowflake connection configuration
            table_name: Name of table containing knowledge base data
            
        Returns:
            DataFrame with knowledge base data
        """
        logger.info(f"Loading knowledge base from Snowflake table {table_name}")
        conn = self.connect_to_snowflake(snowflake_config)
        
        query = f"SELECT domain, company_type, text_content as content FROM {table_name}"
        kb_data = pd.read_sql(query, conn)
        
        conn.close()
        
        return kb_data
    
    def train_with_knowledge_base(self, 
                                 knowledge_base_data: Union[pd.DataFrame, str],
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> 'DomainClassifier':
        """
        Train the classifier using a knowledge base of example companies
        
        Args:
            knowledge_base_data: DataFrame with columns ['domain', 'content', 'company_type'] or path to CSV
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Self (for method chaining)
        """
        # Load data if path is provided
        if isinstance(knowledge_base_data, str):
            knowledge_base_data = self.load_knowledge_base_from_csv(knowledge_base_data)
        
        logger.info(f"Training classifier with {len(knowledge_base_data)} examples")
        logger.info("Company type distribution:")
        for company_type, count in knowledge_base_data['company_type'].value_counts().items():
            logger.info(f"  {company_type}: {count} examples")
        
        # Preprocess content text
        logger.info("Preprocessing text...")
        knowledge_base_data['processed_text'] = knowledge_base_data['content'].apply(self.preprocess_text)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = np.array([
            self.generate_embedding(text) for text in knowledge_base_data['processed_text']
        ])
        
        # Split into train/test sets
        logger.info(f"Splitting data with test_size={test_size}...")
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            knowledge_base_data['company_type'], 
            test_size=test_size, 
            random_state=random_state,
            stratify=knowledge_base_data['company_type']  # Ensure balanced classes in train/test
        )
        
        # Train Random Forest classifier
        logger.info("Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        y_pred = self.classifier.predict(X_test)
        
        logger.info("Classification Report:")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        logger.info("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{cm}")
        
        return self
    
    def classify_domain(self, domain_content: str) -> Dict[str, Any]:
        """
        Classify a domain based on its content
        
        Args:
            domain_content: Text content scraped from the domain
            
        Returns:
            dict: Classification result with confidence scores
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_with_knowledge_base first.")
        
        # Generate embedding for domain content
        embedding = self.generate_embedding(domain_content)
        
        # Get probability scores for each class
        probabilities = self.classifier.predict_proba([embedding])[0]
        
        # Create result dictionary with confidence scores
        result = {
            'predicted_class': self.classifier.classes_[np.argmax(probabilities)],
            'confidence_scores': {}
        }
        
        # Add confidence scores for each company type
        for i, company_type in enumerate(self.classifier.classes_):
            result['confidence_scores'][company_type] = float(probabilities[i])
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model
        """
        if self.classifier is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file
        
        Args:
            filepath: Path to load model from
        """
        logger.info(f"Loading model from {filepath}")
        with open(filepath, 'rb') as f:
            self.classifier = pickle.load(f)
        logger.info("Model loaded successfully")

    def batch_classify_from_snowflake(self, 
                                     snowflake_config: Dict[str, str], 
                                     source_table: str, 
                                     target_table: str) -> None:
        """
        Classify domains from Snowflake source table and save results to target table
        
        Args:
            snowflake_config: Connection config for Snowflake
            source_table: Source table containing domain content
            target_table: Target table to store classification results
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_with_knowledge_base first.")
        
        logger.info(f"Connecting to Snowflake for batch classification")
        conn = self.connect_to_snowflake(snowflake_config)
        cursor = conn.cursor()
        
        # Check if target table exists, create if not
        logger.info(f"Checking if target table {target_table} exists")
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {target_table} (
            ID NUMBER(38,0) NOT NULL autoincrement start 1 increment 1 noorder,
            DOMAIN VARCHAR(255) NOT NULL,
            COMPANY_TYPE VARCHAR(255) NOT NULL,
            CONFIDENCE_SCORE FLOAT NOT NULL,
            ALL_SCORES VARCHAR(1000),
            CLASSIFICATION_DATE TIMESTAMP_NTZ(9) DEFAULT CURRENT_TIMESTAMP(),
            primary key (ID)
        )
        """)
        
        # Fetch domains to classify
        logger.info(f"Fetching domains to classify from {source_table}")
        query = f"SELECT DOMAIN, TEXT_CONTENT FROM {source_table} WHERE TEXT_CONTENT IS NOT NULL"
        cursor.execute(query)
        
        results = []
        domain_count = 0
        start_time = time.time()
        
        logger.info("Starting batch classification")
        for row in cursor:
            domain_count += 1
            domain, content = row
            
            if domain_count % 10 == 0:
                logger.info(f"Classified {domain_count} domains...")
            
            classification = self.classify_domain(content)
            
            # Convert confidence scores to JSON string
            all_scores_json = json.dumps(classification['confidence_scores'])
            
            results.append({
                'domain': domain,
                'company_type': classification['predicted_class'],
                'confidence_score': classification['confidence_scores'][classification['predicted_class']],
                'all_scores': all_scores_json
            })
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Classification completed for {domain_count} domains in {duration:.2f} seconds")
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("No domains to classify or all domains failed classification")
            cursor.close()
            conn.close()
            return
        
        # Save results to Snowflake
        logger.info(f"Saving classification results to {target_table}")
        for _, row in results_df.iterrows():
            # Check if domain already exists in classification table
            check_query = f"SELECT COUNT(*) FROM {target_table} WHERE DOMAIN = %s"
            cursor.execute(check_query, (row['domain'],))
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Update existing record
                update_query = f"""
                UPDATE {target_table}
                SET COMPANY_TYPE = %s,
                    CONFIDENCE_SCORE = %s,
                    ALL_SCORES = %s,
                    CLASSIFICATION_DATE = CURRENT_TIMESTAMP()
                WHERE DOMAIN = %s
                """
                cursor.execute(update_query, (
                    row['company_type'],
                    float(row['confidence_score']),
                    row['all_scores'],
                    row['domain']
                ))
            else:
                # Insert new record
                insert_query = f"""
                INSERT INTO {target_table} (
                    DOMAIN,
                    COMPANY_TYPE,
                    CONFIDENCE_SCORE,
                    ALL_SCORES,
                    CLASSIFICATION_DATE
                ) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP())
                """
                cursor.execute(insert_query, (
                    row['domain'],
                    row['company_type'],
                    float(row['confidence_score']),
                    row['all_scores']
                ))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Results saved to Snowflake successfully")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = DomainClassifier()
    
    # Knowledge base file path (from knowledge_base_builder.py)
    kb_path = "knowledge_base/knowledge_base.csv"
    
    if os.path.exists(kb_path):
        # Train with knowledge base
        classifier.train_with_knowledge_base(kb_path)
        
        # Save model
        classifier.save_model('domain_classifier_model.pkl')
        
        # Example classification
        example_content = "We offer residential home automation, smart home installation, and home theater setup services."
        result = classifier.classify_domain(example_content)
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence scores: {result['confidence_scores']}")
    else:
        print(f"Knowledge base not found at {kb_path}")
        print(f"Please run knowledge_base_builder.py first to create the knowledge base.")
