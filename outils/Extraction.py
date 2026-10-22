import os
import re
import sqlite3
import pandas as pd
import pdfplumber

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
# Gestion dynamique du répertoire racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.getcwd()) == "outils" else os.getcwd()

DOSSIER_CV = os.path.join(BASE_DIR, "data", "cv_pdf")
FICHIER_CSV = os.path.join(BASE_DIR, "data", "dataset_candidats.csv")
FICHIER_EXCEL = os.path.join(BASE_DIR, "data", "dataset_candidats.xlsx")
BDD_SQLITE = os.path.join(BASE_DIR, "recrutement.db")

# ------------------------------------------------------------------------------
# 1. FONCTIONS DE PARSING DES CVS
# ------------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """Extrait le texte brut d'un fichier PDF natif."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ Erreur de lecture sur {os.path.basename(pdf_path)} : {e}")
    return text.strip()


def extract_category(filename):
    """Extrait la catégorie métier depuis le nom du fichier (ex: DATA_SCIENCE_1234.pdf)."""
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[0]
    return "Général"


def extract_identity(text, filename):
    """Extrait une estimation du prénom et nom ou attribue un identifiant."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        words = lines[0].split()
        if len(words) >= 2 and len(words[0]) < 20:
            return words[0].capitalize(), " ".join(words[1:]).upper()
    
    file_id = os.path.splitext(filename)[0]
    return f"Candidat_{file_id}", "Anonyme"


def extract_skills(text):
    """Détecte les compétences clés présentes dans le texte du CV."""
    skills_db = [
        # Tech, IA & Data
        "Python", "SQL", "Machine Learning", "Scikit-Learn", "TensorFlow", 
        "PyTorch", "NLP", "FastAPI", "Flask", "React", "Next.js", "PostgreSQL", 
        "MongoDB", "Docker", "Git", "Power BI", "Excel", "Java", "Spring Boot",
        "C++", "HTML", "CSS", "JavaScript", "Analyse de données", "Deep Learning",
        # Métiers, Gestion & Finance
        "Accounting", "Finance", "HR", "Management", "Marketing", "Sales",
        "Gestion de projet", "Communication", "Recrutement", "Audit", "Budget"
    ]
    found = []
    text_lower = text.lower()
    for skill in skills_db:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return ", ".join(found) if found else "Non spécifiées"


def extract_experience(text):
    """Détermine le niveau d'expérience."""
    text_lower = text.lower()
    exp_pattern = re.search(r'(\d+)\s*(ans|année|années|years|year)\b', text_lower)
    if exp_pattern:
        years = int(exp_pattern.group(1))
        if years < 2: return f"Junior ({years} ans)"
        elif years <= 5: return f"Intermédiaire ({years} ans)"
        else: return f"Sénior ({years} ans)"
        
    if "senior" in text_lower or "lead" in text_lower or "sénior" in text_lower:
        return "Sénior"
    elif "junior" in text_lower or "intern" in text_lower or "débutant" in text_lower:
        return "Junior"
    return "Non précisé"


def extract_education(text):
    """Détecte les diplômes et formations."""
    text_lower = text.lower()
    degrees = []
    mapping = {
        "Doctorat / PhD": ["doctorat", "phd", "doctorate"],
        "Master / Ingénieur": ["master", "ingénieur", "msc", "bachelor of engineering", "bac+5"],
        "Licence / Bachelor": ["licence", "bachelor", "bs", "bsc", "bac+3"],
        "BTS / DUT / Associate": ["bts", "dut", "associate", "bac+2"]
    }
    for level, kws in mapping.items():
        if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in kws):
            degrees.append(level)
            
    return ", ".join(degrees) if degrees else "Formation non détectée"

# ------------------------------------------------------------------------------
# 2. EXÉCUTION DU PIPELINE D'EXTRACTION
# ------------------------------------------------------------------------------

def main():
    if not os.path.exists(DOSSIER_CV):
        print(f"❌ Erreur : Le dossier '{DOSSIER_CV}' est introuvable.")
        print("Assure-toi d'avoir exécuté 'python outils/echantillonner.py' au préalable.")
        return

    fichiers_pdf = [f for f in os.listdir(DOSSIER_CV) if f.lower().endswith('.pdf')]
    print(f"🚀 Début de l'extraction sur {len(fichiers_pdf)} fichiers PDF dans '{DOSSIER_CV}'...\n")

    candidates_list = []

    for idx, file_name in enumerate(fichiers_pdf, start=1):
        pdf_path = os.path.join(DOSSIER_CV, file_name)
        
        # 1. Extraction du texte
        raw_text = extract_text_from_pdf(pdf_path)
        
        # 2. Parsing des champs
        category = extract_category(file_name)
        prenom, nom = extract_identity(raw_text, file_name)
        competences = extract_skills(raw_text)
        exp = extract_experience(raw_text)
        edu = extract_education(raw_text)
        
        candidates_list.append({
            'Fichier_CV': file_name,
            'Categorie_Poste': category,
            'Prenom': prenom,
            'Nom': nom,
            'Niveau_Experience': exp,
            'Formation': edu,
            'Competences': competences,
            'Texte_Brut': raw_text
        })
        
        print(f"[{idx:02d}/{len(fichiers_pdf)}] Extrait : {file_name} | Métier : {category}")

    # Création du DataFrame pandas
    df = pd.DataFrame(candidates_list)

    # 3. Exportation des fichiers CSV / Excel
    os.makedirs(os.path.dirname(FICHIER_CSV), exist_ok=True)
    df.to_csv(FICHIER_CSV, index=False, encoding='utf-8-sig')
    df.to_excel(FICHIER_EXCEL, index=False)
    print(f"\n✅ Fichiers sauvegardés : '{FICHIER_CSV}' et '{FICHIER_EXCEL}'")

    # 4. Insertion en Base de Données SQLite
    conn = sqlite3.connect(BDD_SQLITE)
    df.to_sql("candidats", conn, if_exists="replace", index=False)
    conn.close()
    print(f"💾 Données insérées dans la base SQLite '{BDD_SQLITE}' (table : 'candidats')")

    print("\n🎉 EXTRACTION TERMINÉE AVEC SUCCÈS !")

if __name__ == "__main__":
    main()