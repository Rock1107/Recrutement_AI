import os
import random
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
random.seed(SEED)

# On remonte d'un niveau si le script est exécuté depuis le dossier 'outils'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.getcwd()) == "outils" else os.getcwd()

dossier_archive = os.path.join(BASE_DIR, "archive (4)")
dossier_cible = os.path.join(BASE_DIR, "data", "cv_pdf")

# Nombre de CVs à récupérer par catégorie/métier
TAILLE_ECHANTILLON_PAR_CATEGORIE = 10 

os.makedirs(dossier_cible, exist_ok=True)

print(f"1. Exploration du dossier : '{dossier_archive}'...")

if not os.path.exists(dossier_archive):
    print(f"❌ Erreur : Le dossier '{dossier_archive}' est introuvable.")
    print(f"Vérifie que le dossier 'archive (4)' se trouve bien à la racine : {BASE_DIR}")
else:
    # Recherche récursive de tous les sous-dossiers contenant des fichiers PDF
    categories_trouvees = []

    for root, dirs, files in os.walk(dossier_archive):
        pdfs = [f for f in files if f.lower().endswith('.pdf')]
        if pdfs:
            categories_trouvees.append((root, pdfs))

    print(f"📂 {len(categories_trouvees)} dossier(s) métier détecté(s) avec des fichiers PDF.\n")

    total_copies = 0

    for chemin_cat, fichiers_pdf in categories_trouvees:
        nom_cat = os.path.basename(chemin_cat)
        
        # Sélection aléatoire de N CVs
        echantillon = random.sample(fichiers_pdf, min(len(fichiers_pdf), TAILLE_ECHANTILLON_PAR_CATEGORIE))
        
        print(f"-> Métier '{nom_cat}' : {len(fichiers_pdf)} CVs trouvés. {len(echantillon)} sélectionnés.")
        
        for fichier in echantillon:
            src = os.path.join(chemin_cat, fichier)
            
            # Nom propre avec préfixe du métier sans espaces
            nom_propre = f"{nom_cat}_{fichier}".replace(" ", "_")
            dst = os.path.join(dossier_cible, nom_propre)
            
            shutil.copy(src, dst)
            total_copies += 1

    print(f"\n🎉 Terminé ! Un échantillon total de {total_copies} CVs a été copié dans '{dossier_cible}'.")