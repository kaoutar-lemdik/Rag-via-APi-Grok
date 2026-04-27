import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

load_dotenv()

# ============================================================
#                    CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "Chroma_db")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    else:
        st.error("Erreur : GROQ_API_KEY introuvable dans les secrets Streamlit !")
        st.stop()

MAX_TOKENS = 500
TEMPERATURE = 0.1
TOP_K = 5


# ============================================================
#        CHARGEMENT AVEC CACHE STREAMLIT (CLÉ DU FIX)
# ============================================================

@st.cache_resource
def load_embeddings():
    """Charge le modele d'embeddings UNE SEULE FOIS."""
    print("⏳ Chargement du modèle d'embedding (une seule fois)...")
    emb = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Modèle d'embedding chargé !")
    return emb


@st.cache_resource
def load_database(_embeddings):
    """Connecte la base ChromaDB UNE SEULE FOIS."""
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"Le dossier {DB_DIR} n'existe pas ! "
            f"Lancez d'abord : python ingestion_chroma.py"
        )
    print("⏳ Chargement de ChromaDB...")
    database = Chroma(
        persist_directory=DB_DIR,
        embedding_function=_embeddings
    )
    count = database._collection.count()
    print(f"✅ ChromaDB chargé : {count} chunks")
    return database


@st.cache_resource
def load_llm():
    """Initialise le client Groq UNE SEULE FOIS."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY non trouvée ! "
            "Ajoutez-la dans le fichier .env"
        )
    client = Groq(api_key=GROQ_API_KEY)
    print("✅ Client Groq initialisé !")
    return client


@st.cache_resource
def init_chatbot():
    """Initialise TOUS les composants une seule fois."""
    embeddings = load_embeddings()
    database = load_database(embeddings)
    client = load_llm()
    return embeddings, database, client


# ============================================================
#                    DETECTION DE LANGUE
# ============================================================

def detect_language(text):
    """Detecte si le texte est en arabe ou en francais"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total = len(text.strip())
    if total == 0:
        return "fr"
    return "ar" if arabic_chars > total * 0.3 else "fr"


# ============================================================
#                    RECHERCHE DE DOCUMENTS
# ============================================================

def search_documents(database, query, k=TOP_K):
    """Recherche les documents les plus pertinents dans ChromaDB"""
    results = database.similarity_search("query: " + query, k=k)
    return results


def format_context(documents):
    """Formate les documents recuperes en un seul texte de contexte"""
    context_parts = []
    for i, doc in enumerate(documents):
        text = doc.page_content.replace("passage: ", "")
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Document {i+1} - {source} page {page}]\n{text}"
        )
    return "\n\n".join(context_parts)


# ============================================================
#                    CONSTRUCTION DES MESSAGES
# ============================================================

def build_messages(query, documents):
    """Construit les messages au format chat pour Groq"""
    context = format_context(documents)
    lang = detect_language(query)

    if lang == "ar":
        system_msg = """أنت مساعد خبير في الخزينة العامة للمملكة المغربية.
التعليمات:
- أجب باللغة العربية فقط
- استخدم فقط المعلومات الموجودة في السياق المقدم
- كن مختصراً ودقيقاً
- اذكر مصدر المعلومة عند الإمكان
- إذا لم تجد الإجابة في السياق، قل: لم أجد هذه المعلومة في الوثائق المتاحة
- لا تكرر الإجابة"""

        user_msg = f"""السياق:
{context}

السؤال: {query}"""

    else:
        system_msg = """Vous etes un assistant expert de la TGR (Tresorerie Generale du Royaume du Maroc).
Instructions:
- Repondez directement a la question, sans vous presenter
- Utilisez uniquement les informations du contexte fourni
- Soyez precis et concis
- Citez le document source quand c'est possible
- Si vous ne trouvez pas la reponse, dites: Information non trouvee dans les documents
- Ne repetez pas la reponse
- Ne dites jamais Cordialement ou Merci"""

        user_msg = f"""Contexte:
{context}

Question: {query}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    return messages, lang


# ============================================================
#                    GENERATION DE REPONSE
# ============================================================

def generate_response(client, messages):
    """Genere une reponse avec Groq (LLaMA 3)"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
    )

    text = response.choices[0].message.content.strip()
    return clean_response(text)


def clean_response(text):
    """Nettoie la reponse generee"""
    if not text:
        return "Aucune reponse generee."

    lines = text.split("\n")
    cleaned_lines = []

    stop_phrases = [
        "Note :", "Cordialement", "Merci",
        "[Votre", "Assistant expert", "_____"
    ]

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(phrase) for phrase in stop_phrases):
            break
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()

    if not text:
        return "Information non trouvee dans les documents disponibles."

    return text


# ============================================================
#                    AFFICHAGE DES SOURCES
# ============================================================

def get_sources(documents):
    """Retourne les sources des documents retrouves sans doublons"""
    seen = set()
    sources = []
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})
    return sources


# ============================================================
#                    HISTORIQUE DE CONVERSATION
# ============================================================

def save_conversation(query, response, sources, elapsed, lang):
    """Sauvegarde la conversation dans un fichier log"""
    log_path = os.path.join(BASE_DIR, "conversation_log.txt")
    sources_str = ", ".join(
        [f"{s['source']} (p.{s['page']})" for s in sources]
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Date : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Langue : {lang}\n")
        f.write(f"Question : {query}\n")
        f.write(f"Reponse : {response}\n")
        f.write(f"Sources : {sources_str}\n")
        f.write(f"Temps : {elapsed:.1f}s\n")


# ============================================================
#              FONCTION SIMPLE POUR APP.PY
# ============================================================

def get_rag_response(query: str) -> dict:
    """
    Fonction principale appelée par app.py
    Utilise le cache Streamlit — rapide après le 1er appel.
    
    Returns:
        dict: {"answer": str, "sources": list, "time": float, "lang": str}
    """
    # Charger depuis le cache (instantané après 1er appel)
    _, database, client = init_chatbot()
    
    start = time.time()

    # Recherche dans ChromaDB
    docs = search_documents(database, query)

    if not docs:
        return {
            "answer": "Aucun document pertinent trouvé dans la base de connaissances.",
            "sources": [],
            "time": 0,
            "lang": detect_language(query)
        }

    # Construction des messages
    messages, lang = build_messages(query, docs)

    # Generation de la reponse
    answer = generate_response(client, messages)
    elapsed = time.time() - start

    # Sources
    sources = get_sources(docs)

    # Sauvegarder dans le log
    try:
        save_conversation(query, answer, sources, elapsed, lang)
    except Exception:
        pass

    return {
        "answer": answer,
        "sources": sources,
        "time": elapsed,
        "lang": lang
    }


# ============================================================
#          CLASSE (gardée pour compatibilité terminal)
# ============================================================

class TGRChatbot:
    def __init__(self):
        self.embeddings = None
        self.database = None
        self.client = None
        self.is_loaded = False

    def load(self):
        if self.is_loaded:
            return
        self.embeddings = load_embeddings()
        self.database = load_database(self.embeddings)
        self.client = load_llm()
        self.is_loaded = True

    def ask(self, query):
        if not self.is_loaded:
            self.load()

        start = time.time()
        docs = search_documents(self.database, query)

        if not docs:
            return {
                "answer": "Aucun document pertinent trouvé.",
                "sources": [],
                "time": 0,
                "lang": detect_language(query)
            }

        messages, lang = build_messages(query, docs)
        answer = generate_response(self.client, messages)
        elapsed = time.time() - start
        sources = get_sources(docs)

        try:
            save_conversation(query, answer, sources, elapsed, lang)
        except Exception:
            pass

        return {
            "answer": answer,
            "sources": sources,
            "time": elapsed,
            "lang": lang
        }


# ============================================================
#                    TEST EN TERMINAL
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Agent RAG TGR (Groq) - Test en terminal")
    print("=" * 60)

    chatbot = TGRChatbot()
    chatbot.load()

    print("\n  Pret ! Posez vos questions (quit pour quitter)")
    print("=" * 60)

    while True:
        try:
            query = input("\nVous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("\nAu revoir !")
            break

        result = chatbot.ask(query)

        print(f"\nTGR Agent :")
        print("-" * 40)
        print(result["answer"])
        print("-" * 40)

        if result["sources"]:
            print(f"\nSources :")
            for s in result["sources"]:
                print(f"   -> {s['source']} (page {s['page']})")

        print(f"\nTemps : {result['time']:.1f}s")
