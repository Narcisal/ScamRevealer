# model_engine.py
import os
import joblib
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import config

def jieba_tokenizer(text):
    """Use Jieba to tokenize Chinese text."""
    return jieba.lcut(text)

class FraudModel:
    def __init__(self):
        self.pipeline = None

    def train(self, df_train):
        """Train the Random Forest model pipeline."""
        print("🔄 [Model] Training Random Forest Model...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)),
            ('clf', RandomForestClassifier(n_estimators=config.RF_ESTIMATORS, random_state=config.RANDOM_STATE))
        ])
        self.pipeline.fit(df_train['text'], df_train['label_id'])
        print("✅ [Model] Training Complete.")

    def save(self):
        """Serialize and save the trained model to disk."""
        joblib.dump(self.pipeline, config.MODEL_FILE)
        print(f"💾 [Model] Saved to '{config.MODEL_FILE}'")

    def load(self):
        """
        Attempt to load a saved model from disk.
        Returns: True if successful, False otherwise.
        """
        if os.path.exists(config.MODEL_FILE):
            print(f"💾 [Model] Found saved model: {config.MODEL_FILE}")
            self.pipeline = joblib.load(config.MODEL_FILE)
            print("✅ [Model] Loaded successfully.")
            return True
        return False

    def predict(self, text):
        """
        Perform inference on a single text input.
        Returns: (is_scam: bool, scam_prob: float)
        """
        if not self.pipeline:
            raise Exception("Model not loaded or trained! Call train() or load() first.")
            
        probabilities = self.pipeline.predict_proba([text])[0]
        scam_prob = probabilities[1]
        
        # Threshold 
        is_scam = scam_prob > 0.5
        return is_scam, scam_prob