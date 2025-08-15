import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kinyarwanda_stopwords = [
    "ni", "ya", "na", "mu", "ku", "bya", "yagize", "bari", "uyu", "ibyo",
    "iyo", "kugira", "ubwo", "ubwo", "ibyo", "kandi", "ariko", "cyangwa",
    "nubwo", "ubwo", "icyo", "byose", "ubwo", "ubwo", "ntabwo", "ubwo",
    "yari", "iyo", "uko", "kuba", "ubwo"
]

def preprocess_text(text, stopwords):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def build_tfidf_model(sentences, stopwords):
    preprocessed = [preprocess_text(s, stopwords) for s in sentences]
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(preprocessed)
    return vectorizer, tfidf_matrix

def find_similar_sentences(query, sentences, vectorizer, tfidf_matrix, stopwords, top_n=5):
    query_processed = preprocess_text(query, stopwords)
    query_vec = vectorizer.transform([query_processed])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_n]
    return [(sentences[i], cosine_similarities[i]) for i in top_indices]