import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

print("Starting grader.py")

class BilingualNLPGrader:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.stemmer = PorterStemmer()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.english_stopwords = set(stopwords.words('english'))
        self.kinyarwanda_stopwords = {
            'na', 'mu', 'ku', 'kuri', 'ni', 'cyangwa', 'nk', 'ko', 'kuko',
            'ariko', 'maze', 'naho', 'nyuma', 'mbere', 'hanyuma', 'kandi',
            'cyane', 'cane', 'buri', 'byose', 'byinshi'
        }

    def preprocess_text(self, text, language='en'):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        if language == 'en':
            tokens = [token for token in tokens if token not in self.english_stopwords]
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [token for token in tokens if token not in self.kinyarwanda_stopwords]
        return ' '.join(tokens)

    def extract_features(self, reference_text, student_text, language='en'):
        ref_processed = self.preprocess_text(reference_text, language)
        stu_processed = self.preprocess_text(student_text, language)
        features = {}
        features['length_ratio'] = len(student_text) / max(len(reference_text), 1)
        features['word_count_ratio'] = len(student_text.split()) / max(len(reference_text.split()), 1)
        try:
            if hasattr(self, 'tfidf_fitted') and self.tfidf_fitted:
                tfidf_matrix = self.tfidf_vectorizer.transform([ref_processed, stu_processed])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                features['tfidf_similarity'] = cosine_sim
            else:
                features['tfidf_similarity'] = 0.0
        except:
            features['tfidf_similarity'] = 0.0
        ref_words = set(ref_processed.split())
        stu_words = set(stu_processed.split())
        features['jaccard_similarity'] = len(ref_words & stu_words) / max(len(ref_words | stu_words), 1)
        ref_terms = ref_processed.split()
        stu_terms = stu_processed.split()
        features['term_coverage'] = sum(1 for term in ref_terms if term in stu_terms) / len(ref_terms) if ref_terms else 0.0
        features['semantic_density'] = len(set(stu_terms)) / len(stu_terms) if stu_terms else 0.0
        health_keywords = {
            'en': ['health', 'disease', 'symptom', 'treatment', 'medicine', 'doctor',
                   'patient', 'infection', 'prevention', 'malaria', 'fever', 'nutrition',
                   'bed', 'net', 'mosquito', 'water', 'clinic', 'hospital'],
            'kin': ['ubuzima', 'indwara', 'ibimenyetso', 'ubuvuzi', 'umuganga',
                    'umurwayi', 'kwandura', 'kurinda', 'malariya', 'umuriro', 'intungamubiri',
                    'umuryango', 'ubunyangamugayo', 'amazi', 'ikigo', 'nderabuzima']
        }
        keywords = health_keywords.get(language, health_keywords['en'])
        keyword_count = sum(1 for word in stu_terms if word in keywords)
        features['keyword_density'] = keyword_count / max(len(stu_terms), 1)
        return features

    def train(self, training_data):
        if not training_data:
            raise ValueError("Training data cannot be empty.")
        all_texts = []
        for item in training_data:
            ref_text = self.preprocess_text(item['reference_answer'], item.get('language', 'en'))
            stu_text = self.preprocess_text(item['student_answer'], item.get('language', 'en'))
            all_texts.extend([ref_text, stu_text])
        self.tfidf_vectorizer.fit(all_texts)
        self.tfidf_fitted = True
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
        X = self.scaler.fit_transform(np.array(X))
        y = np.array(y)
        self.model.fit(X, y)
        self.is_trained = True

    def grade(self, reference_answer, student_answer, language='en'):
        if not self.is_trained:
            raise ValueError("Model must be trained before grading.")
        features = self.extract_features(reference_answer, student_answer, language)
        feature_vector = np.array([list(features.values())])
        feature_scaled = self.scaler.transform(feature_vector)
        predicted_score = self.model.predict(feature_scaled)[0]
        final_score = max(1, min(5, round(predicted_score)))
        confidence = min(1.0, features['tfidf_similarity'] + features['jaccard_similarity'])
        feedback = self.generate_feedback(features, final_score, language)
        return {
            'score': final_score,
            'confidence': confidence,
            'features': features,
            'feedback': feedback
        }

    def generate_feedback(self, features, score, language='en'):
        feedback_templates = {
            'en': {
                5: "Excellent answer. You've covered the key points clearly.",
                4: "Good answer. Consider adding a few more details.",
                3: "Fair answer. Try to include more key terms and details.",
                2: "Your answer needs more explanation and examples.",
                1: "This answer is too brief or off-topic. Please review the material."
            },
            'kin': {
                5: "Igisubizo cyiza cyane. Wasobanuye ingingo zose neza.",
                4: "Igisubizo cyiza. Wagerageza kongeramo andi makuru.",
                3: "Hari ibyo wasobanura neza kurushaho.",
                2: "Ibisubizo bikwiye kuba birambuye kurushaho.",
                1: "Igisubizo ni kigufi cyangwa kidahuye n'ibikenewe."
            }
        }
        templates = feedback_templates.get(language, feedback_templates['en'])
        base_feedback = templates.get(score, templates[3])
        if features['term_coverage'] < 0.5:
            if language == 'en':
                base_feedback += " Try to mention more important terms from the reference."
            else:
                base_feedback += " Gerageza gushyiramo amagambo y'ingenzi yifashishijwe mu gisubizo cy'ukuri."
        return base_feedback
    

   

if __name__ == "__main__":
    # Sample training data
    print("Training grader...")
    training_data = [
        {
            'reference_answer': "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment.",
            'student_answer': "To prevent malaria, we should use treated bed nets every night, spray homes with insecticide, remove standing water, use repellent creams, and go to health center when we have fever.",
            'score': 5,
            'language': 'en'
        },
        {
            'reference_answer': "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment.",
            'student_answer': "Use bed nets and remove standing water to prevent malaria.",
            'score': 3,
            'language': 'en'
        }
    ]
    print("Training grader...")

    grader = BilingualNLPGrader()
    grader.train(training_data)

    # Test grading
    reference = "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment."
    student = "We should use bed nets and remove water to prevent malaria."
    result = grader.grade(reference, student, language='en')

    print("Score:", result['score'])
    print("Confidence:", result['confidence'])
    print("Feedback:", result['feedback'])
    print("Features:", result['features'])