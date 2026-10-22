import os
import re
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

STOPWORDS = set([
    "le", "la", "les", "un", "une", "des", "du", "de", "en", "et", "est", "que", "qui", 
    "dans", "sur", "pour", "avec", "par", "au", "aux", "ce", "ces", "pas", "plus",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from",
    "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "this"
])

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text_clean = text.lower()
    text_clean = re.sub(r'http\S+|www\S+|\S+@\S+', '', text_clean)
    text_clean = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœa-z]', ' ', text_clean)
    tokens = [w for w in text_clean.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

def get_experience_level_numeric(text):
    if not isinstance(text, str):
        return 0
    t = text.lower()
    if any(k in t for k in ["sénior", "senior", "lead", "10 ans", "15 ans", "18 ans", "20 ans", "7 ans", "13 ans"]):
        return 3
    elif any(k in t for k in ["intermédiaire", "confirmé", "5 ans", "3 ans", "4 ans", "2 ans"]):
        return 2
    elif any(k in t for k in ["junior", "débutant", "intern"]):
        return 1
    return 0

def get_education_level_numeric(text):
    if not isinstance(text, str):
        return 0
    t = text.lower()
    if any(k in t for k in ["doctorat", "phd"]):
        return 4
    elif any(k in t for k in ["master", "ingénieur", "bac+5"]):
        return 3
    elif any(k in t for k in ["licence", "bachelor", "bac+3"]):
        return 2
    elif any(k in t for k in ["bts", "dut", "associate", "bac+2"]):
        return 1
    return 0

_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _sbert_model

def run_matching_dataframe(df_cand, df_off, id_offre, top_n=10, strict_mode=False):
    """
    Exécute le matching en mémoire directement à partir de DataFrames.
    """
    df_cand = df_cand.copy()
    if 'Texte_Nettoye' not in df_cand.columns:
        df_cand['Texte_Nettoye'] = df_cand['Texte_Brut'].apply(preprocess_text)

    offre = df_off[df_off['ID_Offre'] == id_offre]
    if offre.empty:
        raise ValueError(f"Offre '{id_offre}' introuvable.")

    row_off = offre.iloc[0]
    exp_exigee = str(row_off.get('Niveau_Experience_Exige', ''))
    edu_exigee = str(row_off.get('Formation_Exigee', ''))
    texte_offre = (str(row_off.get('Description_Brute', '')) + " " + str(row_off.get('Competences_Exigees', ''))).lower()

    num_exp_exigee = get_experience_level_numeric(exp_exigee)
    num_edu_exigee = get_education_level_numeric(edu_exigee)

    # NLP
    corpus = [texte_offre] + df_cand['Texte_Nettoye'].tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    scores_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    sbert_model = get_sbert_model()
    embedding_offre = sbert_model.encode([texte_offre])
    embeddings_candidats = sbert_model.encode(df_cand['Texte_Nettoye'].tolist(), show_progress_bar=False)
    scores_sbert = cosine_similarity(embedding_offre, embeddings_candidats).flatten()

    df_res = df_cand.copy()
    df_res['Score_TFIDF (%)'] = np.round(scores_tfidf * 100, 2)
    df_res['Score_SBERT (%)'] = np.round(scores_sbert * 100, 2)
    df_res['Score_Sémantique (%)'] = np.round(0.3 * df_res['Score_TFIDF (%)'] + 0.7 * df_res['Score_SBERT (%)'], 2)

    df_res['Num_Exp'] = df_res['Niveau_Experience'].apply(get_experience_level_numeric)
    df_res['Num_Edu'] = df_res['Formation'].apply(get_education_level_numeric)

    if strict_mode:
        df_res = df_res[(df_res['Num_Exp'] >= num_exp_exigee) & (df_res['Num_Edu'] >= num_edu_exigee)]
        df_res['Score_Final (%)'] = df_res['Score_Sémantique (%)']
    else:
        def compute_business_score(row):
            score = row['Score_Sémantique (%)']
            if row['Num_Exp'] >= num_exp_exigee:
                score += 5.0
            elif 0 < row['Num_Exp'] < num_exp_exigee:
                score -= 10.0
                
            if row['Num_Edu'] >= num_edu_exigee:
                score += 5.0
            elif 0 < row['Num_Edu'] < num_edu_exigee:
                score -= 5.0
                
            return np.clip(np.round(score, 2), 0, 100)

        df_res['Score_Final (%)'] = df_res.apply(compute_business_score, axis=1)

    df_res = df_res.sort_values(by='Score_Final (%)', ascending=False).reset_index(drop=True)
    df_res['Rang'] = df_res.index + 1

    cols_show = [
        'Rang', 'Fichier_CV', 'Prenom', 'Nom', 'Categorie_Poste', 
        'Niveau_Experience', 'Formation', 'Competences', 
        'Score_TFIDF (%)', 'Score_SBERT (%)', 'Score_Final (%)'
    ]
    cols_exist = [c for c in cols_show if c in df_res.columns]
    return df_res[cols_exist].head(top_n)

def run_matching(id_offre, db_path="recrutement.db", top_n=10, strict_mode=False):
    conn = sqlite3.connect(db_path)
    df_cand = pd.read_sql("SELECT * FROM candidats", conn)
    df_off = pd.read_sql("SELECT * FROM offres", conn)
    conn.close()
    return run_matching_dataframe(df_cand, df_off, id_offre, top_n, strict_mode)
