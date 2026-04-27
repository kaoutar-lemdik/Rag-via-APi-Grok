import os
import streamlit as st
import time
from chat_rag_groq import TGRChatbot
from questionnaire import (
    afficher_questionnaire,
    verifier_reponses,
    sauvegarder_google_sheets
)
import ingestion_chroma  # Importe ton script d'ingestion

# ============================================================
#                    CONFIGURATION PAGE
# ============================================================

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "Chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "Data_test")

# --- INITIALISATION AUTOMATIQUE ---
# Si le fichier de base de données n'existe pas ou si le dossier est vide
if not os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
    st.warning("⚠️ Base de connaissances introuvable. Initialisation en cours...")
    
    # On s'assure que le dossier Data_test existe et n'est pas vide
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        with st.spinner("Analyse des documents TGR... Patientez quelques instants."):
            try:
                # Appelle la fonction principale de ton script d'ingestion
                ingestion_chroma.main() 
                st.success("✅ Intelligence TGR prête !")
                st.rerun() # Relance l'app pour charger la nouvelle base
            except Exception as e:
                st.error(f"Erreur lors de l'ingestion : {e}")
    else:
        st.error("❌ Le dossier 'Data_test' est vide. Ajoutez des PDF sur GitHub !")
        st.stop()

st.set_page_config(
    page_title="Assistant TGR — Chatbot IA",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
#                    SESSION STATE
# ============================================================

if "etape" not in st.session_state:
    st.session_state.etape = "accueil"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "nb_messages" not in st.session_state:
    st.session_state.nb_messages = 0
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

MIN_ECHANGES = 3

QUESTIONS_SUGGEREES = [
    "Quels sont les services proposés par la TGR en ligne ?",
    "Comment payer ma taxe d'habitation en ligne ?",
    "Quels documents fournir pour obtenir une attestation fiscale ?",
]

# ============================================================
#                    STYLES CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton > button {
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#                    PAGE 1 : ACCUEIL
# ============================================================

def page_accueil():

    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Assistant Virtuel — TGR</h1>
        <h3>Trésorerie Générale du Royaume du Maroc</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    **Bienvenue !** 👋

    Dans le cadre d'une recherche universitaire, nous avons développé un 
    **prototype de chatbot intelligent** capable de répondre à vos questions 
    sur les services de la Trésorerie Générale du Royaume.

    ### Votre participation en 2 étapes simples :

    | Étape | Quoi | Durée |
    |-------|------|-------|
    | 1️⃣ | **Tester le chatbot** en posant quelques questions | ~3 min |
    | 2️⃣ | **Donner votre avis** via un court questionnaire | ~3 min |

    ---

    ⚠️ **Important :**
    - Ce chatbot est un **prototype de recherche**, pas un service officiel
    - Les réponses ne sauraient engager la responsabilité de la TGR
    - Votre participation est **100% anonyme** et **volontaire**
    """)

    st.markdown("---")

    consentement = st.checkbox(
        "✅ J'ai lu les informations ci-dessus et j'accepte de participer.",
        key="consentement"
    )

    if consentement:
        if st.button(
            " Commencer",
            type="primary",
            use_container_width=True
        ):
            st.session_state.etape = "chatbot"
            st.rerun()


# ============================================================
#                    PAGE 2 : CHATBOT
# ============================================================

def page_chatbot():

    # Charger le chatbot
    if st.session_state.chatbot is None:
        with st.spinner("⏳ Chargement du chatbot... (peut prendre quelques secondes)"):
            try:
                chatbot = TGRChatbot()
                chatbot.load()
                st.session_state.chatbot = chatbot
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")
                return

    chatbot = st.session_state.chatbot

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown("## 📌 Guide")

        nb = st.session_state.nb_messages

        if nb < MIN_ECHANGES:
            st.warning(f"💬 Échanges : **{nb} / {MIN_ECHANGES}**")
            st.progress(nb / MIN_ECHANGES)
            st.markdown(
                f"Posez encore **{MIN_ECHANGES - nb}** question(s) "
                f"avant de pouvoir donner votre avis."
            )
        else:
            st.success(f"✅ **{nb}** échanges réalisés !")
            st.progress(1.0)
            st.markdown("Vous pouvez continuer ou passer au questionnaire.")

            if st.button(
                "📋 Donner mon avis",
                type="primary",
                use_container_width=True,
                key="btn_sidebar"
            ):
                st.session_state.etape = "questionnaire"
                st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Essayez ces questions :")

        for i, question in enumerate(QUESTIONS_SUGGEREES):
            if st.button(
                f"📌 {question}",
                key=f"suggestion_{i}",
                use_container_width=True
            ):
                st.session_state.pending_question = question
                st.rerun()

        st.markdown("---")
        st.markdown("🏛️ **TGR** | 🤖 LLaMA 3 via Groq")
        st.markdown("🇫🇷 Français | 🇲🇦 Arabe")

    # ---- ZONE DE CHAT ----
    st.markdown("## 💬 Chatbot TGR")

    # Message d'accueil
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant", avatar="🏛️"):
            st.markdown("""
            👋 **Bonjour !** Je suis l'assistant virtuel de la **TGR**.

            Je peux vous renseigner sur :
            - 📋 Les services et procédures de la TGR
            - 💳 Les modalités de paiement en ligne
            - 📄 Les documents nécessaires pour vos démarches

            **Posez votre question** ci-dessous ou cliquez sur 
            une suggestion dans le menu à gauche ! 👈
            """)

    # Afficher l'historique
    for msg in st.session_state.messages:
        avatar = "🙋" if msg["role"] == "user" else "🏛️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"- {s['source']} (page {s['page']})"
                            )
                if msg.get("time"):
                    st.caption(f"⏱️ {msg['time']:.1f}s")

    # Traiter une question suggérée
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        traiter_question(chatbot, question)

    # Champ de saisie
    if prompt := st.chat_input("Posez votre question ici..."):
        traiter_question(chatbot, prompt)

    # Bouton questionnaire en bas
    if st.session_state.nb_messages >= MIN_ECHANGES:
        st.markdown("---")
        st.success(
            "✅ Merci d'avoir testé le chatbot ! "
            "Vous pouvez continuer ou passer au questionnaire."
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "📋 Donner mon avis maintenant",
                type="primary",
                use_container_width=True,
                key="btn_bottom"
            ):
                st.session_state.etape = "questionnaire"
                st.rerun()


def traiter_question(chatbot, question):
    """Traite une question de l'utilisateur."""

    # Afficher la question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user", avatar="🙋"):
        st.markdown(question)

    # Générer la réponse
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("🔍 Recherche en cours..."):
            try:
                result = chatbot.ask(question)
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                return

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📎 Sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"- {s['source']} (page {s['page']})"
                    )

        st.caption(f"⏱️ {result['time']:.1f}s")

    # Sauvegarder dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "time": result["time"]
    })

    st.session_state.nb_messages += 1


# ============================================================
#                    PAGE 3 : QUESTIONNAIRE
# ============================================================

def page_questionnaire():

    reponses = afficher_questionnaire()

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "✅ Soumettre mes réponses",
            type="primary",
            use_container_width=True
        ):
            valide, message = verifier_reponses(reponses)

            if not valide:
                st.error(f"⚠️ {message}")
            else:
                with st.spinner("📤 Envoi en cours..."):
                    try:
                        sauvegarder_google_sheets(reponses)
                        st.session_state.etape = "merci"
                        st.rerun()
                    except Exception as e:
                        st.error(
                            f"❌ Erreur lors de la sauvegarde : {e}"
                        )


# ============================================================
#                    PAGE 4 : MERCI
# ============================================================

def page_merci():

    st.markdown("""
    <div class="main-header">
        <h1>🎉 Merci pour votre participation !</h1>
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

    st.markdown("""
    Votre contribution est **précieuse** et nous aidera à mieux 
    comprendre le potentiel de l'intelligence artificielle au service 
    des usagers du service public marocain.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "💬 Questions posées",
            st.session_state.nb_messages
        )
    with col2:
        st.metric("⏱️ Durée estimée", "~6 min")

    st.markdown("---")

    st.markdown("""
    ### 🙏 Aidez-nous !
    
    Partagez ce lien avec vos proches pour nous aider 
    à collecter plus de réponses :
    """)

    st.code("https://rag-via-api-grok-3qs88yvxyufljb9zbhrndk.streamlit.app/", language=None)

    st.markdown("""
    ---
    📧 Contact : **[kaoutar.lemdik@usms.ac.ma]**
    """)


# ============================================================
#                    ROUTEUR PRINCIPAL
# ============================================================

def main():
    etape = st.session_state.etape

    if etape == "accueil":
        page_accueil()
    elif etape == "chatbot":
        page_chatbot()
    elif etape == "questionnaire":
        page_questionnaire()
    elif etape == "merci":
        page_merci()


if __name__ == "__main__":
    main()
