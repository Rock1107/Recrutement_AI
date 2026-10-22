# ==============================================================================
# MODULE DE MATCHING CV - OFFRES D'EMPLOI (TF-IDF + SENTENCE-BERT + FILTRAGE MÉTIER)
# Projet de Mémoire : Moteur de Recommandation RH Intelligent
# ==============================================================================

import os
import re
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# 1. PRÉTRAITEMENT TEXTUEL NLP
# ------------------------------------------------------------------------------
STOPWORDS = set([
    "le", "la", "les", "un", "une", "des", "du", "de", "en", "et", "est", "que", "qui", 
    "dans", "sur", "pour", "avec", "par", "au", "aux", "ce", "ces", "pas", "plus",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from",
    "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "this"
])

def preprocess_text(text: str) -> str:
    """
    Nettoie le texte brut (passage en minuscules, suppression des e-mails, URLs,
    caractères spéciaux et stopwords).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text_clean = text.lower()
    text_clean = re.sub(r'http\S+|www\S+|\S+@\S+', '', text_clean)
    text_clean = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœa-z]', ' ', text_clean)
    tokens = [w for w in text_clean.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

# ------------------------------------------------------------------------------
# 2. EVALUATION NUMÉRIQUE DES CRITÈRES MÉTIER
# ------------------------------------------------------------------------------
def get_experience_level_numeric(text: str) -> int:
    """Convertit le niveau d'expérience textuel en échelle numérique (0 à 3)."""
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

def get_education_level_numeric(text: str) -> int:
    """Convertit le niveau de formation textuel en échelle numérique (0 à 4)."""
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

# ------------------------------------------------------------------------------
# 3. ACCÈS BASE DE DONNÉES ET CHARGEMENT DU MODÈLE SBERT
# ------------------------------------------------------------------------------
def load_database_data(db_path: str = "recrutement.db"):
    """Charge les tables 'candidats' et 'offres' depuis SQLite."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base de données introuvable : {db_path}")
    conn = sqlite3.connect(db_path)
    df_cand = pd.read_sql("SELECT * FROM candidats", conn)
    df_off = pd.read_sql("SELECT * FROM offres", conn)
    conn.close()
    return df_cand, df_off

# Instance globale mise en cache pour éviter de recharger SBERT à chaque appel
_sbert_model = None

def get_sbert_model(model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    """Charge et réutilise le modèle Sentence-BERT."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model

# ------------------------------------------------------------------------------
# 4. MOTEUR DE MATCHING PRINCIPAL
# ------------------------------------------------------------------------------
def run_matching(id_offre: str, db_path: str = "recrutement.db", top_n: int = 10, strict_mode: bool = False):
    """
    Calcule le matching hybride (TF-IDF + SBERT) et applique les règles métier RH.

    Args:
        id_offre (str): Identifiant de l'offre (ex: 'OFFRE_01', 'OFFRE_02').
        db_path (str): Chemin d'accès à la base SQLite 'recrutement.db'.
        top_n (int): Nombre de candidats à retourner.
        strict_mode (bool): 
            - False: Soft Filter (Pondéré avec bonus/pénalités métier).
            - True: Hard Filter (Exclusion stricte des candidats sous-qualifiés).

    Returns:
        pd.DataFrame: Tableau structuré des candidats classés par pertinence.
    """
    df_cand, df_off = load_database_data(db_path)

    if 'Texte_Nettoye' not in df_cand.columns:
        df_cand['Texte_Nettoye'] = df_cand['Texte_Brut'].apply(preprocess_text)

    offre = df_off[df_off['ID_Offre'] == id_offre]
    if offre.empty:
        raise ValueError(f"L'offre '{id_offre}' n'existe pas dans la base de données.")

    row_off = offre.iloc[0]
    exp_exigee = row_off['Niveau_Experience_Exige']
    edu_exigee = row_off['Formation_Exigee']
    texte_offre = (str(row_off['Description_Brute']) + " " + str(row_off['Competences_Exigees'])).lower()

    num_exp_exigee = get_experience_level_numeric(exp_exigee)
    num_edu_exigee = get_education_level_numeric(edu_exigee)

    # A. Vectorisation TF-IDF (30%)
    corpus = [texte_offre] + df_cand['Texte_Nettoye'].tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    scores_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # B. Vectorisation SBERT (70%)
    sbert_model = get_sbert_model()
    embedding_offre = sbert_model.encode([texte_offre])
    embeddings_candidats = sbert_model.encode(df_cand['Texte_Nettoye'].tolist(), show_progress_bar=False)
    scores_sbert = cosine_similarity(embedding_offre, embeddings_candidats).flatten()

    # C. Combinaison des scores NLP
    df_res = df_cand.copy()
    df_res['Score_TFIDF (%)'] = np.round(scores_tfidf * 100, 2)
    df_res['Score_SBERT (%)'] = np.round(scores_sbert * 100, 2)
    df_res['Score_Sémantique (%)'] = np.round(0.3 * df_res['Score_TFIDF (%)'] + 0.7 * df_res['Score_SBERT (%)'], 2)

    df_res['Num_Exp'] = df_res['Niveau_Experience'].apply(get_experience_level_numeric)
    df_res['Num_Edu'] = df_res['Formation'].apply(get_education_level_numeric)

    # D. Application de la logique métier (Hard Filter vs Soft Filter)
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

    # E. Tri et préparation de la restitution
    df_res = df_res.sort_values(by='Score_Final (%)', ascending=False).reset_index(drop=True)
    df_res['Rang'] = df_res.index + 1

    cols_show = [
        'Rang', 'Fichier_CV', 'Prenom', 'Nom', 'Categorie_Poste', 
        'Niveau_Experience', 'Formation', 'Competences', 
        'Score_TFIDF (%)', 'Score_SBERT (%)', 'Score_Sémantique (%)', 'Score_Final (%)'
    ]
    return df_res[cols_show].head(top_n)

if __name__ == '__main__':
    print("🧪 Test d'exécution autonome du module 'src/matching.py'...")
    df_test = run_matching(id_offre='OFFRE_01', top_n=5, strict_mode=False)
    print(df_test.to_string())
