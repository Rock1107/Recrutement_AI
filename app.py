import os
import re
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pypdf import PdfReader

# ------------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE & STYLES CSS SUR-MESURE
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartRecruit AI - Dashboard Dynamique",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalise
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #f8FAFA;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #FB5563;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background-color: #E2E0;
        border: 1px solid #E2E0;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div.stButton > button[type="primary"] {
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

try:
    from src.matching import run_matching_dataframe, preprocess_text
except ImportError:
    st.error("❌ Le module `src/matching.py` est introuvable.")
    st.stop()

# INITIALISATION DU SESSION STATE
if 'custom_offres' not in st.session_state:
    st.session_state['custom_offres'] = []

if 'custom_candidats' not in st.session_state:
    st.session_state['custom_candidats'] = []

# EN-TÊTE
st.markdown('<div class="main-title">🤖 RECRUTEMENT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Importez vos propres offres d\'emploi et téléversez des CVs PDF pour exécuter le matching IA et générer votre shortlist.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# BARRE LATÉRALE
# ------------------------------------------------------------------------------
st.sidebar.header("📁 Source des Données")
data_source = st.sidebar.radio(
    "Source d'analyse :",
    options=["Données Personnalisées (Import Temps Réel)", "Base de Démonstration (SQLite)"]
)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Filtrage & Matching RH")

filter_mode = st.sidebar.radio(
    "Mode d'évaluation :",
    options=["Soft Filter (Pondéré avec bonus/pénalités)", "Hard Filter (Strict - Exclusion)"],
    help="• Soft Filter : Ajuste le score avec bonus/pénalités.\n• Hard Filter : Élimine les candidats sous-qualifiés."
)
is_strict = "Hard Filter" in filter_mode

st.sidebar.markdown("---")
st.sidebar.header("🏆 Option Shortlist")
shortlist_size = st.sidebar.selectbox(
    "Taille de la Shortlist (Top N) :",
    options=[3, 5, 10, 15],
    index=1,
    help="Nombre de meilleurs candidats à retenir dans la shortlist."
)

top_n = st.sidebar.slider("Nombre total de candidats à afficher :", min_value=1, max_value=20, value=10)

# ------------------------------------------------------------------------------
# SECTION 1 : GESTION DES OFFRES ET CVS PERSONNALISÉS
# ------------------------------------------------------------------------------
if data_source == "Données Personnalisées (Import Temps Réel)":
    st.markdown("---")
    st.subheader("📥 Zone d'Importation & Saisie")
    
    tab_offres, tab_cvs = st.tabs(["📝 Ajouter une Offre d'Emploi", "📄 Téléverser des CVs (PDF)"])
    
    with tab_offres:
        with st.form("form_nouvelle_offre", clear_on_submit=True):
            col_id, col_titre = st.columns([1, 3])
            id_offre_input = col_id.text_input("ID Offre (ex: OFFRE_PERSO_01)", value=f"OFFRE_CUSTOM_{len(st.session_state['custom_offres'])+1}")
            titre_offre_input = col_titre.text_input("Titre du Poste", placeholder="ex: Développeur Python Senior")
            
            col_cat, col_exp, col_edu = st.columns(3)
            cat_input = col_cat.text_input("Catégorie Métier", placeholder="ex: IT, DATA_SCIENCE, HR")
            exp_input = col_exp.selectbox("Niveau d'expérience exigé", ["Junior (0 à 2 ans)", "Intermédiaire (2 à 5 ans)", "Sénior (5 ans et +)"])
            edu_input = col_edu.selectbox("Formation exigée", ["BTS / DUT / Bac+2", "Licence / Bachelor / Bac+3", "Master / Ingénieur / Bac+5", "Doctorat / PhD"])
            
            comp_input = st.text_input("Compétences clés exigées (séparées par des virgules)", placeholder="ex: Python, SQL, Docker, FastAPI")
            desc_input = st.text_area("Description complète de la fiche de poste", placeholder="Saisissez ou collez les détails du poste...")
            
            submit_offre = st.form_submit_button("➕ Ajouter l'Offre à la Session")
            if submit_offre and titre_offre_input and desc_input:
                st.session_state['custom_offres'].append({
                    'ID_Offre': id_offre_input,
                    'Titre_Poste': titre_offre_input,
                    'Categorie_Metier': cat_input if cat_input else 'GENERAL',
                    'Niveau_Experience_Exige': exp_input,
                    'Formation_Exigee': edu_input,
                    'Competences_Exigees': comp_input,
                    'Description_Brute': desc_input
                })
                st.success(f"✅ Offre '{titre_offre_input}' ajoutée avec succès !")

    with tab_cvs:
        uploaded_files = st.file_uploader("Glissez-déposez un ou plusieurs CVs au format PDF :", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("⚙️ Extraire & Analyser les CVs téléversés", type="primary"):
                with st.spinner("Extraction du texte et analyse des profils en cours..."):
                    new_cands = []
                    for pdf_file in uploaded_files:
                        try:
                            reader = PdfReader(pdf_file)
                            raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
                            cleaned_text = preprocess_text(raw_text)
                            
                            exp_detec = "Sénior" if "senior" in raw_text.lower() or "10 ans" in raw_text.lower() else "Intermédiaire"
                            edu_detec = "Master / Ingénieur" if "master" in raw_text.lower() or "ingénieur" in raw_text.lower() else "Licence / Bachelor"
                            
                            new_cands.append({
                                'Fichier_CV': pdf_file.name,
                                'Prenom': 'Candidat',
                                'Nom': pdf_file.name.replace('.pdf', ''),
                                'Categorie_Poste': 'DÉPOSÉ',
                                'Niveau_Experience': exp_detec,
                                'Formation': edu_detec,
                                'Competences': 'Extraites du PDF',
                                'Texte_Brut': raw_text,
                                'Texte_Nettoye': cleaned_text
                            })
                        except Exception as e:
                            st.error(f"Erreur lors de la lecture de {pdf_file.name}: {e}")
                    
                    st.session_state['custom_candidats'].extend(new_cands)
                    st.success(f"🎉 {len(new_cands)} CV(s) extrait(s) et ajoutés à la session !")

# ------------------------------------------------------------------------------
# SECTION 2 : EXECUTION DU MATCHING & TABLEAU DE BORD
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Moteur de Matching & Analyse Résultats")

# Préparation des DataFrames
if data_source == "Données Personnalisées (Import Temps Réel)":
    if not st.session_state['custom_offres']:
        st.info("💡 Veuillez ajouter au moins une offre personnalisée ci-dessus pour démarrer.")
        st.stop()
    if not st.session_state['custom_candidats']:
        st.info("💡 Veuillez téléverser au moins un CV PDF ci-dessus pour lancer la comparaison.")
        st.stop()
        
    df_offres = pd.DataFrame(st.session_state['custom_offres'])
    df_candidats = pd.DataFrame(st.session_state['custom_candidats'])
else:
    DB_PATH = "recrutement.db"
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Base de données `{DB_PATH}` introuvable à la racine du projet.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    df_offres = pd.read_sql("SELECT * FROM offres", conn)
    df_candidats = pd.read_sql("SELECT * FROM candidats", conn)
    conn.close()

# Sélection de l'offre
offres_dict = {f"[{row['ID_Offre']}] {row['Titre_Poste']}": row['ID_Offre'] for _, row in df_offres.iterrows()}
selected_label = st.sidebar.selectbox("Offre à évaluer :", list(offres_dict.keys()))
selected_id_offre = offres_dict[selected_label]

# Affichage des détails
offre_info = df_offres[df_offres['ID_Offre'] == selected_id_offre].iloc[0]
with st.expander(f"📋 **Fiche descriptive : {offre_info['Titre_Poste']}**", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ID Offre", offre_info['ID_Offre'])
    c2.metric("Domaine", offre_info.get('Categorie_Metier', 'N/A'))
    c3.metric("Expérience Exigée", offre_info.get('Niveau_Experience_Exige', 'N/A'))
    c4.metric("Formation Exigée", offre_info.get('Formation_Exigee', 'N/A'))
    st.markdown(f"**Compétences clés :** `{offre_info.get('Competences_Exigees', '')}`")

# CALCUL DU MATCHING
with st.spinner("⏳ Calcul des similarités vectorielles (TF-IDF & Sentence-BERT)..."):
    try:
        df_leaderboard = run_matching_dataframe(
            df_cand=df_candidats,
            df_off=df_offres,
            id_offre=selected_id_offre,
            top_n=top_n,
            strict_mode=is_strict
        )
    except Exception as e:
        st.error(f"Erreur lors du calcul du matching : {e}")
        st.stop()

if df_leaderboard.empty:
    st.warning("⚠️ Aucun candidat ne satisfait les critères stricts en mode Hard Filter.")
else:
    actual_shortlist_size = min(shortlist_size, len(df_leaderboard))
    
    tab_global, tab_shortlist = st.tabs([
        "📊 Classement & Analyse Globale", 
        f"⭐ Shortlist des Candidats (Top {actual_shortlist_size})"
    ])

    # ONGLET 1 : CLASSEMENT GLOBAL
    with tab_global:
        top_cand = df_leaderboard.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🥇 Meilleur Match", f"{top_cand['Nom']}")
        m2.metric("Score SBERT", f"{top_cand['Score_SBERT (%)']}%")
        m3.metric("Score TF-IDF", f"{top_cand['Score_TFIDF (%)']}%")
        m4.metric("Score Final Métier", f"{top_cand['Score_Final (%)']}%")

        st.markdown("---")
        st.subheader("🏆 Classement complet")

        st.dataframe(
            df_leaderboard,
            column_config={
                "Rang": st.column_config.NumberColumn("Rang", format="#%d"),
                "Score_Final (%)": st.column_config.ProgressColumn(
                    "Score Final Métier (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                ),
                "Score_SBERT (%)": st.column_config.NumberColumn(format="%.2f %%"),
                "Score_TFIDF (%)": st.column_config.NumberColumn(format="%.2f %%"),
            },
            use_container_width=True,
            hide_index=True
        )

        csv_data = df_leaderboard.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Télécharger le Classement Complet (CSV)",
            data=csv_data,
            file_name=f"classement_complet_{selected_id_offre}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("📈 Comparaison Graphique des Scores")
        fig_bar = px.bar(
            df_leaderboard,
            x='Fichier_CV',
            y=['Score_TFIDF (%)', 'Score_SBERT (%)', 'Score_Final (%)'],
            barmode='group',
            title="Décomposition des Scores par Candidat",
            labels={'value': 'Score (%)', 'variable': 'Métrique'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ONGLET 2 : SHORTLIST
    with tab_shortlist:
        df_shortlist = df_leaderboard.head(actual_shortlist_size).copy()

        st.markdown(f"### ⭐ Shortlist Sélectionnée : {actual_shortlist_size} Candidat(s) Retenu(s)")
        st.write("Voici la liste restreinte des profils à convoquer en priorité pour les entretiens.")

        csv_shortlist = df_shortlist.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⭐ Télécharger la Shortlist (CSV)",
            data=csv_shortlist,
            file_name=f"shortlist_{selected_id_offre}.csv",
            mime="text/csv",
            key="dl_shortlist"
        )

        st.markdown("---")

        for _, candidate in df_shortlist.iterrows():
            rang_val = int(candidate.get('Rang', 1))
            nom_val = candidate.get('Nom', 'Candidat')
            score_final_val = candidate.get('Score_Final (%)', 0.0)

            with st.expander(f"🏅 **#{rang_val} — {nom_val}** | Score Final : **{score_final_val:.2f} %**"):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Score Mots-Clés (TF-IDF)", f"{candidate.get('Score_TFIDF (%)', 0):.2f} %")
                sc2.metric("Score Sémantique (SBERT)", f"{candidate.get('Score_SBERT (%)', 0):.2f} %")
                sc3.metric("Score Final Métier", f"{score_final_val:.2f} %")

                st.markdown(f"**Fichier CV :** `{candidate.get('Fichier_CV', 'N/A')}`")
                st.markdown(f"**Niveau d'expérience :** {candidate.get('Niveau_Experience', 'N/A')}")
                st.markdown(f"**Formation :** {candidate.get('Formation', 'N/A')}")
                
                match_cand = df_candidats[df_candidats['Fichier_CV'] == candidate.get('Fichier_CV')]
                if not match_cand.empty and 'Texte_Brut' in match_cand.columns:
                    raw_cv_text = match_cand.iloc[0]['Texte_Brut']
                    st.markdown("#### 📄 Contenu Extrait du CV")
                    st.text_area(
                        "Texte intégral",
                        value=raw_cv_text,
                        height=220,
                        key=f"txt_{selected_id_offre}_{rang_val}",
                        disabled=True
                    )
                    st.download_button(
                        label=f"📥 Télécharger le CV texte de {nom_val}",
                        data=raw_cv_text,
                        file_name=f"CV_{nom_val}.txt",
                        mime="text/plain",
                        key=f"dl_txt_{selected_id_offre}_{rang_val}"
                    )

st.markdown("---")
st.caption("SmartRecruit AI — Solution Dynamique de Recrutement Intelligente")