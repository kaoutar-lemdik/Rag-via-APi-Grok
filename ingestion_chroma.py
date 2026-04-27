import os
import re
import shutil
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================================
#                    CONFIGURATION GÉNÉRALE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "Data_test")    
DB_DIR = os.path.join(BASE_DIR, "Chroma_db")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".docx": "DOCX",
}
WARN_EXTENSIONS = {
    ".doc": " Format .doc non supporté → Convertissez en .docx ou .pdf"
}


# ============================================================
#                    INITIALISATION EMBEDDINGS
# ============================================================

def init_embeddings():
    """Initialise le modèle d'embeddings multilingual-e5 sur CPU"""
    print(f"\n Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
    print(f"   (Device: CPU — Quadro P520 2Go VRAM insuffisante)")

    start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    elapsed = time.time() - start
    print(f" Modèle chargé en {elapsed:.1f}s\n")
    return embeddings


# ============================================================
#                    NETTOYAGE DU TEXTE
# ============================================================

def clean_text(text):
    """
    Nettoie le texte extrait des PDF/DOCX.
    Gère les caractères parasites courants dans les documents arabes et français.
    """
    if not text:
        return ""

    # Suppression des caractères nuls et BOM
    text = text.replace("\x00", "")
    text = text.replace("\ufeff", "")

    # Suppression des caractères de contrôle (sauf \n et \t)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalisation des espaces multiples
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalisation des sauts de ligne multiples (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Suppression des lignes vides contenant uniquement des espaces
    text = re.sub(r'\n\s+\n', '\n\n', text)

    return text.strip()


# ============================================================
#                    CHARGEMENT DES DOCUMENTS
# ============================================================

def load_single_file(path, filename):
    """Charge un seul fichier et retourne les documents extraits"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"   PDF  : {filename} → {len(docs)} page(s)")
        return docs

    elif ext == ".docx":
        loader = Docx2txtLoader(path)
        docs = loader.load()
        print(f"   DOCX : {filename} → {len(docs)} section(s)")
        return docs

    elif ext in WARN_EXTENSIONS:
        print(f"  {WARN_EXTENSIONS[ext]} : {filename}")
        return []

    else:
        print(f"    Ignoré (non supporté) : {filename}")
        return []


def load_documents(directory):
    """Charge tous les documents d'un répertoire"""
    if not os.path.exists(directory):
        print(f" Le dossier '{directory}' n'existe pas !")
        return []

    files = sorted(os.listdir(directory))
    if not files:
        print(f" Le dossier '{directory}' est vide !")
        return []

    print(f" Fichiers trouvés : {len(files)}\n")

    all_docs = []
    success_count = 0
    error_count = 0

    for filename in files:
        filepath = os.path.join(directory, filename)

        # Ignorer les dossiers
        if os.path.isdir(filepath):
            continue

        try:
            docs = load_single_file(filepath, filename)
            if docs:
                all_docs.extend(docs)
                success_count += 1
        except Exception as e:
            print(f"   ERREUR sur {filename}: {e}")
            error_count += 1

    print(f"\n Bilan chargement :")
    print(f"    Fichiers chargés : {success_count}")
    print(f"    Erreurs          : {error_count}")
    print(f"    Pages/Sections   : {len(all_docs)}")

    return all_docs


# ============================================================
#                    DÉCOUPAGE (CHUNKING)
# ============================================================

def split_documents(docs):
    """
    Découpe les documents en chunks avec des séparateurs
    adaptés aux textes arabes et français.
    """
    # Séparateurs ordonnés du plus prioritaire au moins prioritaire
    # Inclut les séparateurs arabes (،  ؟  !)
    separators = [
        "\n\n",      # Double saut de ligne (nouveau paragraphe)
        "\n",        # Saut de ligne simple
        ".",         # Fin de phrase (français)
        "。",        # Fin de phrase (autres langues)
        "؟",         # Point d'interrogation arabe
        "!",         # Point d'exclamation
        "،",         # Virgule arabe
        ",",         # Virgule française
        " ",         # Espace
        ""           # Dernier recours : caractère par caractère
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )

    chunks = text_splitter.split_documents(docs)
    return chunks


# ============================================================
#                    PRÉPARATION POUR E5
# ============================================================

def prepare_chunks_for_e5(chunks):
    """
    Prépare les chunks pour le modèle multilingual-e5 :
    - Nettoie le texte
    - Ajoute le préfixe 'passage: ' obligatoire pour E5
    - Filtre les chunks vides
    """
    prepared = []
    empty_count = 0

    for chunk in chunks:
        cleaned = clean_text(chunk.page_content)

        # Ignorer les chunks trop courts (< 20 caractères)
        if len(cleaned) < 20:
            empty_count += 1
            continue

        chunk.page_content = "passage: " + cleaned
        prepared.append(chunk)

    if empty_count > 0:
        print(f"     {empty_count} chunk(s) trop court(s) supprimé(s)")

    return prepared


# ============================================================
#                    GESTION DE CHROMADB
# ============================================================

def reset_database(db_path):
    """Supprime l'ancienne base ChromaDB pour éviter les doublons"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"  Ancienne base supprimée : {db_path}")
    else:
        print(f" Nouvelle base sera créée : {db_path}")



def store_in_chromadb(chunks, embeddings, db_path):
    """Stocke les chunks dans ChromaDB par lots de 500 pour éviter les erreurs de batch"""
    print(f"\n Enregistrement de {len(chunks)} chunks dans ChromaDB...")
    
    # Découper la liste des chunks par tranches de 500
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"   Envoi du batch {i // batch_size + 1} ({len(batch)} chunks)...")
        
        Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=db_path
        )
    
    print(" Enregistrement terminé avec succès !")


# ============================================================
#                    VÉRIFICATION & APERÇU
# ============================================================

def show_preview(chunks, num_samples=3):
    """Affiche un aperçu des premiers chunks pour vérification"""
    print(f"\n{'─'*50}")
    print(f" APERÇU DES CHUNKS (premiers {min(num_samples, len(chunks))})")
    print(f"{'─'*50}")

    for i, chunk in enumerate(chunks[:num_samples]):
        source = chunk.metadata.get('source', 'inconnu')
        filename = os.path.basename(source)
        page = chunk.metadata.get('page', '?')

        print(f"\n🔹 Chunk {i+1} | Fichier: {filename} | Page: {page}")
        print(f"   Taille: {len(chunk.page_content)} caractères")
        # Afficher les 150 premiers caractères
        preview = chunk.page_content[:150]
        print(f"   Contenu: {preview}...")


def verify_database(db_path, embeddings):
    """Vérifie que la base ChromaDB fonctionne correctement"""
    print(f"\n{'─'*50}")
    print(f" VÉRIFICATION DE LA BASE")
    print(f"{'─'*50}")

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # Test avec une requête simple en français
    test_query_fr = "query: Quelles sont les missions de la TGR ?"
    results_fr = vectorstore.similarity_search(test_query_fr, k=2)

    print(f"\n🇫🇷 Test FR: '{test_query_fr}'")
    if results_fr:
        for i, doc in enumerate(results_fr):
            print(f"   Résultat {i+1}: {doc.page_content[:100]}...")
    else:
        print("    Aucun résultat trouvé")

    # Test avec une requête en arabe
    test_query_ar = "query: ما هي مهام الخزينة العامة للمملكة؟"
    results_ar = vectorstore.similarity_search(test_query_ar, k=2)

    print(f"\n🇲🇦 Test AR: '{test_query_ar}'")
    if results_ar:
        for i, doc in enumerate(results_ar):
            print(f"   Résultat {i+1}: {doc.page_content[:100]}...")
    else:
        print("    Aucun résultat trouvé")


# ============================================================
#                    POINT D'ENTRÉE PRINCIPAL
# ============================================================

def main():
    print(f"\n{'='*60}")
    print(f"    RAG TGR — Pipeline d'Ingestion")
    print(f"   Source  : {SOURCE_DIR}")
    print(f"   Base    : {DB_DIR}")
    print(f"   Modèle : {EMBEDDING_MODEL}")
    print(f"    Chunks : {CHUNK_SIZE} chars | overlap {CHUNK_OVERLAP}")
    print(f"{'='*60}")

    total_start = time.time()

    # Étape 1 : Initialiser les embeddings
    embeddings = init_embeddings()

    # Étape 2 : Réinitialiser la base
    reset_database(DB_DIR)

    # Étape 3 : Charger les documents
    print(f"\n{'─'*50}")
    print(f" ÉTAPE 1 : CHARGEMENT DES DOCUMENTS")
    print(f"{'─'*50}\n")
    raw_docs = load_documents(SOURCE_DIR)

    if not raw_docs:
        print("\n Aucun document chargé. Vérifiez le dossier source.")
        return

    # Étape 4 : Découpage
    print(f"\n{'─'*50}")
    print(f"  ÉTAPE 2 : DÉCOUPAGE EN CHUNKS")
    print(f"{'─'*50}")
    chunks = split_documents(raw_docs)
    print(f"   Chunks bruts créés : {len(chunks)}")

    # Étape 5 : Nettoyage + préparation E5
    chunks = prepare_chunks_for_e5(chunks)
    print(f"   Chunks finaux      : {len(chunks)}")

    # Étape 6 : Aperçu
    show_preview(chunks)

    # Étape 7 : Stockage ChromaDB
    print(f"\n{'─'*50}")
    print(f" ÉTAPE 3 : STOCKAGE DANS CHROMADB")
    print(f"{'─'*50}")
    store_in_chromadb(chunks, embeddings, DB_DIR)

    # Étape 8 : Vérification
    verify_database(DB_DIR, embeddings)

    # Résumé final
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"   INGESTION TERMINÉE AVEC SUCCÈS")
    print(f"   Documents traités : {len(raw_docs)}")
    print(f"    Chunks stockés    : {len(chunks)}")
    print(f"    Temps total       : {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()