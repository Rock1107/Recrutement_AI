import pymupdf as fitz  # PyMuPDF

def extract_pdf_text(pdf_file):
    """
    Extrait tout le texte brut d'un fichier PDF (objet fichier Streamlit ou chemin local).
    """
    text = ""
    try:
        # Si c'est un fichier uploadé via Streamlit ou un chemin de fichier
        with fitz.open(stream=pdf_file.read() if hasattr(pdf_file, 'read') else pdf_file, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Erreur lors de l'extraction du PDF : {e}")
    
    return text