from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def calculate_matching_score(job_description, cv_texts):
    """
    Calcule un score de matching hybride (TF-IDF + Sentence-BERT) 
    entre la description d'une offre d'emploi et une liste de CVs.
    """
    if not cv_texts:
        return [], [], []

    # 1. Approche lexicale : TF-IDF (Mots-clés stricts)
    documents = [job_description] + cv_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    job_tfidf = tfidf_matrix[0:1]
    cv_tfidf = tfidf_matrix[1:]
    
    # Similarité cosinus pour TF-IDF (ramenée en pourcentage %)
    tfidf_scores = cosine_similarity(job_tfidf, cv_tfidf).flatten() * 100

    # 2. Approche sémantique : Sentence-BERT (Compréhension du sens)
    # Utilisation du modèle léger et performant all-MiniLM-L6-v2
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    job_embedding = model.encode([job_description])
    cv_embeddings = model.encode(cv_texts)
    
    # Similarité cosinus pour SBERT (ramenée en pourcentage %)
    sbert_scores = cosine_similarity(job_embedding, cv_embeddings).flatten() * 100

    # 3. Score hybride combiné (30% Lexical + 70% Sémantique)
    final_scores = (0.3 * tfidf_scores) + (0.7 * sbert_scores)

    return tfidf_scores, sbert_scores, final_scores