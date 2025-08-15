import numpy as np
import re
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedBilingualNLPGrader:
    def __init__(self, model_path=None):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased from 5000
            stop_words='english',
            ngram_range=(1, 3),  # Added trigrams
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        self.stemmer = PorterStemmer()
        self.model = RandomForestRegressor(
            n_estimators=200,  # Increased from 100
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.english_stopwords = set(stopwords.words('english'))
        
        # Enhanced Kinyarwanda stopwords
        self.kinyarwanda_stopwords = {
            'na', 'mu', 'ku', 'kuri', 'ni', 'cyangwa', 'nk', 'ko', 'kuko',
            'ariko', 'maze', 'naho', 'nyuma', 'mbere', 'hanyuma', 'kandi',
            'cyane', 'cane', 'buri', 'byose', 'byinshi', 'ya', 'bya', 'yagize',
            'bari', 'uyu', 'ibyo', 'iyo', 'kugira', 'ubwo', 'icyo', 'ntabwo',
            'yari', 'uko', 'kuba', 'nka', 'nk', 'nko', 'nkuko', 'nkuko',
            'nkuko', 'nkuko', 'nkuko', 'nkuko', 'nkuko', 'nkuko', 'nkuko'
        }
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def preprocess_text(self, text, language='en'):
        """Enhanced text preprocessing with better error handling"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Language-specific processing
        if language == 'en':
            tokens = [token for token in tokens if token not in self.english_stopwords and len(token) > 2]
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:  # Kinyarwanda
            tokens = [token for token in tokens if token not in self.kinyarwanda_stopwords and len(token) > 2]
        
        return ' '.join(tokens)

    def extract_features(self, reference_text, student_text, language='en'):
        """Enhanced feature extraction with more sophisticated metrics"""
        ref_processed = self.preprocess_text(reference_text, language)
        stu_processed = self.preprocess_text(student_text, language)
        
        features = {}
        
        # Basic ratios
        features['length_ratio'] = len(student_text) / max(len(reference_text), 1)
        features['word_count_ratio'] = len(student_text.split()) / max(len(reference_text.split()), 1)
        
        # TF-IDF similarity
        try:
            if hasattr(self, 'tfidf_fitted') and self.tfidf_fitted:
                tfidf_matrix = self.tfidf_vectorizer.transform([ref_processed, stu_processed])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                features['tfidf_similarity'] = cosine_sim
            else:
                features['tfidf_similarity'] = 0.0
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            features['tfidf_similarity'] = 0.0
        
        # Set-based similarities
        ref_words = set(ref_processed.split())
        stu_words = set(stu_processed.split())
        
        features['jaccard_similarity'] = len(ref_words & stu_words) / max(len(ref_words | stu_words), 1)
        features['overlap_ratio'] = len(ref_words & stu_words) / max(len(ref_words), 1)
        
        # Term coverage and density
        ref_terms = ref_processed.split()
        stu_terms = stu_processed.split()
        
        features['term_coverage'] = sum(1 for term in ref_terms if term in stu_terms) / len(ref_terms) if ref_terms else 0.0
        features['semantic_density'] = len(set(stu_terms)) / len(stu_terms) if stu_terms else 0.0
        
        # Enhanced keyword analysis
        health_keywords = {
            'en': [
                'health', 'disease', 'symptom', 'treatment', 'medicine', 'doctor',
                'patient', 'infection', 'prevention', 'malaria', 'fever', 'nutrition',
                'bed', 'net', 'mosquito', 'water', 'clinic', 'hospital', 'vaccine',
                'hygiene', 'sanitation', 'pregnancy', 'childbirth', 'breastfeeding',
                'diarrhea', 'pneumonia', 'vaccination', 'immunization', 'antibiotic'
            ],
            'kin': [
                'ubuzima', 'indwara', 'ibimenyetso', 'ubuvuzi', 'umuganga',
                'umurwayi', 'kwandura', 'kurinda', 'malariya', 'umuriro', 'intungamubiri',
                'umuryango', 'ubunyangamugayo', 'amazi', 'ikigo', 'nderabuzima',
                'ubworozi', 'ubwishingizi', 'ubuvuzi', 'ubuzima', 'indwara',
                'ibimenyetso', 'ubuvuzi', 'umuganga', 'umurwayi', 'kwandura'
            ]
        }
        
        keywords = health_keywords.get(language, health_keywords['en'])
        keyword_count = sum(1 for word in stu_terms if word in keywords)
        features['keyword_density'] = keyword_count / max(len(stu_terms), 1)
        features['keyword_coverage'] = keyword_count / max(len([w for w in ref_terms if w in keywords]), 1) if any(w in keywords for w in ref_terms) else 0.0
        
        # Additional features
        features['avg_word_length'] = np.mean([len(word) for word in stu_terms]) if stu_terms else 0.0
        features['sentence_complexity'] = len(stu_terms) / max(len(student_text.split()), 1)
        
        return features

    def train(self, training_data, validation_split=0.2):
        """Enhanced training with validation and model evaluation"""
        if not training_data:
            raise ValueError("Training data cannot be empty.")
        
        logger.info(f"Training model with {len(training_data)} samples")
        
        # Prepare all texts for TF-IDF fitting
        all_texts = []
        for item in training_data:
            ref_text = self.preprocess_text(item['reference_answer'], item.get('language', 'en'))
            stu_text = self.preprocess_text(item['student_answer'], item.get('language', 'en'))
            all_texts.extend([ref_text, stu_text])
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(all_texts)
        self.tfidf_fitted = True
        
        # Extract features
        X = []
        y = []
        for item in training_data:
            features = self.extract_features(
                item['reference_answer'],
                item['student_answer'],
                item.get('language', 'en')
            )
            X.append(list(features.values()))
            y.append(item['score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data if validation_split > 0
        if validation_split > 0 and len(X) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        logger.info(f"Training R² score: {train_score:.3f}")
        logger.info(f"Validation R² score: {val_score:.3f}")
        
        # Cross-validation if enough data
        if len(X) > 5:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=min(5, len(X)//2))
            logger.info(f"Cross-validation R² scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.is_trained = True
        return {'train_score': train_score, 'val_score': val_score}

    def grade(self, reference_answer, student_answer, language='en'):
        """Enhanced grading with better error handling and confidence calculation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before grading.")
        
        try:
            features = self.extract_features(reference_answer, student_answer, language)
            feature_vector = np.array([list(features.values())])
            feature_scaled = self.scaler.transform(feature_vector)
            
            predicted_score = self.model.predict(feature_scaled)[0]
            final_score = max(1, min(5, round(predicted_score)))
            
            # Enhanced confidence calculation
            confidence_factors = [
                features['tfidf_similarity'],
                features['jaccard_similarity'],
                features['term_coverage'],
                features['keyword_density']
            ]
            confidence = np.mean(confidence_factors)
            
            feedback = self.generate_feedback(features, final_score, language)
            
            return {
                'score': final_score,
                'confidence': confidence,
                'features': features,
                'feedback': feedback,
                'raw_score': predicted_score
            }
        except Exception as e:
            logger.error(f"Error during grading: {e}")
            raise

    def generate_feedback(self, features, score, language='en'):
        """Enhanced feedback generation with more specific guidance"""
        feedback_templates = {
            'en': {
                5: "Excellent answer! You've comprehensively covered all key points with clear explanations.",
                4: "Good answer. You've covered most important points. Consider adding a few more details for completeness.",
                3: "Fair answer. You've mentioned some key points but could include more specific details and terminology.",
                2: "Your answer needs more explanation and examples. Try to include more key terms and concepts.",
                1: "This answer is too brief or off-topic. Please review the material and provide more comprehensive coverage."
            },
            'kin': {
                5: "Igisubizo cyiza cyane! Wasobanuye ingingo zose neza kandi urambuye.",
                4: "Igisubizo cyiza. Wasobanuye ingingo nyamukuru. Wagerageza kongeramo andi makuru.",
                3: "Hari ibyo wasobanura neza kurushaho. Wagerageza kongeramo amagambo y'ingenzi.",
                2: "Ibisubizo bikwiye kuba birambuye kurushaho. Gerageza gushyiramo amagambo y'ingenzi.",
                1: "Igisubizo ni kigufi cyangwa kidahuye n'ibikenewe. Subiramo ibikenewe."
            }
        }
        
        templates = feedback_templates.get(language, feedback_templates['en'])
        base_feedback = templates.get(score, templates[3])
        
        # Add specific feedback based on features
        if features['term_coverage'] < 0.5:
            if language == 'en':
                base_feedback += " Try to mention more important terms from the reference answer."
            else:
                base_feedback += " Gerageza gushyiramo amagambo y'ingenzi yifashishijwe mu gisubizo cy'ukuri."
        
        if features['keyword_density'] < 0.1:
            if language == 'en':
                base_feedback += " Include more health-related terminology in your answer."
            else:
                base_feedback += " Shyiramo amagambo yerekeye ubuzima mu gisubizo."
        
        return base_feedback

    def save_model(self, filepath):
        """Save the trained model and components"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'tfidf_fitted': getattr(self, 'tfidf_fitted', False)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model and components"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.tfidf_fitted = model_data.get('tfidf_fitted', False)
        
        logger.info(f"Model loaded from {filepath}")

    def evaluate_model(self, test_data):
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        X_test = []
        y_test = []
        
        for item in test_data:
            features = self.extract_features(
                item['reference_answer'],
                item['student_answer'],
                item.get('language', 'en')
            )
            X_test.append(list(features.values()))
            y_test.append(item['score'])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
