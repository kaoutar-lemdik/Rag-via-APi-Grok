import os
import re
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ============================================================
#                    CONFIGURATION
# ============================================================

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR     = os.path.join(BASE_DIR, "Data_test")
PINECONE_INDEX = "tgr-index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50

# ============================================================
#                    EMBEDDINGS
# ============================================================

def init_embeddings():
    print(f"🔄 Chargement du modèle : {EMBEDDING_MODEL}")
    start = time.time()
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            'device': 'cpu',
            'local_files_only': True
        },
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"✅ Modèle chargé en {time.time() - start:.1f}s")
    return embeddings


# ============================================================
#                    NETTOYAGE TEXTE
# ============================================================

def clean_text(text):
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s+\n', '\n\n', text)
    return text.strip()


# ============================================================
#                    CHARGEMENT DOCUMENTS
# ============================================================

def load_documents(directory):
    if not os.path.exists(directory):
        print(f"❌ Dossier introuvable : {directory}")
        return []

    files = sorted(os.listdir(directory))
    print(f"📂 Fichiers trouvés : {len(files)}\n")

    all_docs      = []
    success_count = 0
    error_count   = 0

    for filename in files:
        filepath = os.path.join(directory, filename)

        if os.path.isdir(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
                docs   = loader.load()
                print(f"   ✅ PDF  : {filename} → {len(docs)} page(s)")
                all_docs.extend(docs)
                success_count += 1

            elif ext == ".docx":
                loader = Docx2txtLoader(filepath)
                docs   = loader.load()
                print(f"   ✅ DOCX : {filename} → {len(docs)} section(s)")
                all_docs.extend(docs)
                success_count += 1

            elif ext == ".doc":
                print(f"   ⚠️  DOC ignoré : {filename}")

            else:
                print(f"   ⏭️  Ignoré : {filename}")

        except Exception as e:
            print(f"   ❌ ERREUR sur {filename}: {e}")
            error_count += 1

    print(f"\n📊 Bilan : {success_count} chargés | {error_count} erreurs | {len(all_docs)} pages")
    return all_docs


# ============================================================
#                    DÉCOUPAGE EN CHUNKS
# ============================================================

def split_documents(docs):
    separators = ["\n\n", "\n", ".", "؟", "!", "،", ",", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )
    return splitter.split_documents(docs)


# ============================================================
#                    PREPARATION E5
# ============================================================

def prepare_chunks_for_e5(chunks):
    prepared    = []
    empty_count = 0

    for chunk in chunks:
        cleaned = clean_text(chunk.page_content)

        if len(cleaned) < 20:
            empty_count += 1
            continue

        chunk.page_content = "passage: " + cleaned
        prepared.append(chunk)

    if empty_count > 0:
        print(f"   ⚠️ {empty_count} chunk(s) trop courts supprimés")

    return prepared


# ============================================================
#                    ENVOI VERS PINECONE
# ============================================================

def send_to_pinecone(chunks, embeddings):
    api_key = os.environ.get("PINECONE_API_KEY")

    print(f"\n🚀 Connexion à Pinecone...")
    pc      = Pinecone(api_key=api_key)
    indexes = [i.name for i in pc.list_indexes().indexes]

    if PINECONE_INDEX not in indexes:
        print(f"❌ Index '{PINECONE_INDEX}' introuvable !")
        print(f"   Index disponibles : {indexes}")
        return False

    print(f"✅ Index '{PINECONE_INDEX}' trouvé !")
    print(f"📤 Envoi de {len(chunks)} chunks vers Pinecone...")
    print(f"⏳ Cela peut prendre 5-10 minutes...\n")

    batch_size    = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch     = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        print(f"   📦 Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        texts     = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]
        ids       = [f"chunk_{i + j}" for j in range(len(batch))]

        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=PINECONE_INDEX,
            metadatas=metadatas,
            ids=ids,
            pinecone_api_key=api_key
        )

    print(f"\n✅ Envoi terminé avec succès !")
    return True


# ============================================================
#                    MAIN
# ============================================================

def main():
    print(f"\n{'='*60}")
    print(f"    RAG TGR — Ingestion vers PINECONE")
    print(f"{'='*60}\n")

    total_start = time.time()

    # Étape 1 : Embeddings
    embeddings = init_embeddings()

    # Étape 2 : Chargement
    print(f"\n{'─'*50}")
    print(f" ÉTAPE 1 : CHARGEMENT DES DOCUMENTS")
    print(f"{'─'*50}\n")
    raw_docs = load_documents(SOURCE_DIR)

    if not raw_docs:
        print("❌ Aucun document chargé !")
        return

    # Étape 3 : Découpage
    print(f"\n{'─'*50}")
    print(f" ÉTAPE 2 : DÉCOUPAGE EN CHUNKS")
    print(f"{'─'*50}")
    chunks = split_documents(raw_docs)
    print(f"✂️  Chunks bruts : {len(chunks)}")

    # Étape 4 : Préparation E5
    chunks = prepare_chunks_for_e5(chunks)
    print(f"✅ Chunks finaux : {len(chunks)}")

    # Étape 5 : Envoi Pinecone
    print(f"\n{'─'*50}")
    print(f" ÉTAPE 3 : ENVOI VERS PINECONE")
    print(f"{'─'*50}")
    success = send_to_pinecone(chunks, embeddings)

    # Résumé
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    if success:
        print(f"✅ INGESTION TERMINÉE AVEC SUCCÈS !")
    else:
        print(f"❌ INGESTION ÉCHOUÉE !")
    print(f"   Chunks envoyés : {len(chunks)}")
    print(f"   Temps total    : {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()