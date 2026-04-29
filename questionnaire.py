import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import csv
import json
import os

# ============================================================
#                    CONFIGURATION
# ============================================================

LIKERT_OPTIONS = {
    1: "Pas du tout d'accord",
    2: "Pas d'accord",
    3: "Neutre",
    4: "D'accord",
    5: "Tout à fait d'accord"
}

# ============================================================
#                    PROFIL SOCIODÉMOGRAPHIQUE
# ============================================================

PROFIL_QUESTIONS = {
    "genre": {
        "label": "Votre genre",
        "options": ["Homme", "Femme"]
    },
    "age": {
        "label": "Votre tranche d'âge",
        "options": [
            "18-25 ans",
            "26-35 ans",
            "36-45 ans",
            "46-55 ans",
            "Plus de 55 ans"
        ]
    },
    "education": {
        "label": "Votre niveau d'éducation",
        "options": [
            "Baccalauréat ou moins",
            "Bac+2 / Bac+3 (DUT, Licence)",
            "Bac+5 (Master, Ingénieur)",
            "Doctorat ou plus"
        ]
    },
    "familiarite_ia": {
        "label": "Avez-vous déjà utilisé un chatbot ou assistant IA (ChatGPT, Siri, etc.) ?",
        "options": [
            "Jamais",
            "Une ou deux fois",
            "Occasionnellement",
            "Régulièrement"
        ]
    }
}

# ============================================================
#                    ITEMS LIKERT
# ============================================================

QUESTIONNAIRE_SECTIONS = {
    "qualite_percue": {
        "titre": "📊 Qualité perçue des réponses du chatbot",
        "consigne": "Évaluez la qualité des réponses que vous avez reçues du chatbot.",
        "items": {
            "QP1": "Les réponses fournies par le chatbot étaient pertinentes par rapport à mes questions.",
            "QP3": "Les informations fournies me semblaient correctes et fiables.",
            "QP5": "Les réponses étaient suffisamment détaillées et complètes.",
            "QP6": "Le langage utilisé par le chatbot était clair et compréhensible.",
            "QP7": "Le chatbot a répondu rapidement à mes questions."
        }
    },
    "satisfaction": {
        "titre": "😊 Satisfaction globale",
        "consigne": "Évaluez votre niveau de satisfaction suite à votre interaction.",
        "items": {
            "SAT1": "Je suis globalement satisfait(e) de mon interaction avec le chatbot.",
            "SAT2": "L'expérience avec le chatbot a répondu à mes attentes.",
            "SAT3": "L'interaction avec le chatbot a été une expérience positive."
        }
    },
    "confiance": {
        "titre": "🏛️ Confiance institutionnelle envers la TGR",
        "consigne": "Évaluez votre perception de la TGR après cette interaction.",
        "items": {
            "CI1": "Après cette interaction, je perçois la TGR comme une institution compétente.",
            "CI3": "Je fais confiance à la TGR pour fournir des informations fiables et exactes.",
            "CI5": "J'ai le sentiment que la TGR se soucie véritablement des besoins de ses usagers."
        }
    },
    "utilite_percue": {
        "titre": "💡 Utilité perçue de l'IA",
        "consigne": "Évaluez votre perception de l'utilité de l'IA dans le service public.",
        "items": {
            "UP1": "L'utilisation d'un chatbot IA est utile pour obtenir des informations sur les services publics.",
            "UP2": "Un chatbot IA peut me permettre de gagner du temps dans mes démarches administratives.",
            "UP3": "Un chatbot IA améliore mon accès à l'information par rapport aux canaux traditionnels."
        }
    },
    "facilite_percue": {
        "titre": "🖱️ Facilité d'utilisation",
        "consigne": "Évaluez la facilité d'utilisation du chatbot.",
        "items": {
            "FU1": "L'interaction avec le chatbot était facile et intuitive.",
            "FU2": "Je n'ai pas eu besoin d'effort particulier pour utiliser le chatbot."
        }
    }
}


# ============================================================
#                    FONCTIONS D'AFFICHAGE
# ============================================================

def afficher_profil():
    """Affiche et collecte les questions sociodémographiques."""

    st.markdown("### 👤 Votre profil")
    st.markdown("*Quelques informations pour mieux connaître votre profil.*")
    st.markdown("---")

    reponses = {}

    for key, q in PROFIL_QUESTIONS.items():
        reponses[key] = st.selectbox(
            q["label"],
            options=["— Sélectionnez —"] + q["options"],
            key=f"profil_{key}"
        )

    return reponses


def afficher_likert(code, texte):
    """Affiche un item Likert avec des boutons radio horizontaux."""

    st.markdown(f"**{code}.** {texte}")

    reponse = st.radio(
        label=code,
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} — {LIKERT_OPTIONS[x]}",
        horizontal=True,
        key=f"likert_{code}",
        label_visibility="collapsed"
    )

    st.markdown("---")

    return reponse


def afficher_questionnaire():
    """Affiche le questionnaire complet et retourne toutes les réponses."""

    st.markdown("## 📋 Donnez-nous votre avis !")

    st.markdown("""
    Merci d'avoir testé le chatbot ! 🙏
    
    Répondez aux questions suivantes pour nous aider à évaluer 
    cette expérience. **Il n'y a pas de bonne ou de mauvaise réponse.**
    Seule votre perception compte.
    
    ⏱️ Durée : **4-5 minutes**
    """)

    st.markdown("---")

    toutes_reponses = {}

    # ---- PARTIE 1 : PROFIL ----
    profil = afficher_profil()
    toutes_reponses.update(profil)

    st.markdown("---")

    # ---- PARTIES 2 à 6 : ITEMS LIKERT ----
    for section_key, section in QUESTIONNAIRE_SECTIONS.items():
        st.markdown(f"### {section['titre']}")
        st.markdown(f"*{section['consigne']}*")
        st.markdown("---")

        for code, texte in section["items"].items():
            reponse = afficher_likert(code, texte)
            toutes_reponses[code] = reponse

    # ---- COMMENTAIRE LIBRE ----
    st.markdown("### 💬 Commentaire libre (facultatif)")
    commentaire = st.text_area(
        "Si vous souhaitez partager une remarque sur votre expérience :",
        key="commentaire_libre",
        height=100,
        placeholder="Votre commentaire ici..."
    )
    toutes_reponses["commentaire"] = commentaire

    return toutes_reponses


# ============================================================
#                    VÉRIFICATION DES RÉPONSES
# ============================================================

def verifier_reponses(reponses):
    """Vérifie que toutes les réponses obligatoires sont remplies."""

    for key in PROFIL_QUESTIONS.keys():
        if reponses.get(key) == "— Sélectionnez —":
            return False, f"Veuillez remplir le champ : {PROFIL_QUESTIONS[key]['label']}"

    return True, ""


# ============================================================
#                    SAUVEGARDE — GOOGLE SHEETS
# ============================================================

def sauvegarder_google_sheets(reponses):
    """
    Sauvegarde les réponses dans Google Sheets.
    Si Google Sheets n'est pas configuré, sauvegarde en CSV local.
    """

    # Ajouter les métadonnées
    reponses["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reponses["nb_echanges"] = st.session_state.get("nb_messages", 0)

    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        # Option 1 : Credentials depuis Streamlit Secrets (déploiement cloud)
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(
                creds_dict, scopes=scope
            )
        # Option 2 : Credentials depuis un fichier local (développement)
        elif os.path.exists("google_credentials.json"):
            creds = Credentials.from_service_account_file(
                "google_credentials.json", scopes=scope
            )
        # Option 3 : Pas de credentials → sauvegarde CSV
        else:
            return sauvegarder_csv(reponses)

        # Connexion à Google Sheets
        client = gspread.authorize(creds)
        sheet = client.open("Resultats_PFE_TGR").sheet1

        # Ajouter les en-têtes si la feuille est vide
        existing = sheet.get_all_values()
        if len(existing) == 0:
            sheet.append_row(list(reponses.keys()))

        # Ajouter la ligne de données
        sheet.append_row(list(reponses.values()))

        return True

    except Exception as e:
        st.warning(
            f"⚠️ Google Sheets non disponible ({e}). "
            f"Sauvegarde locale en CSV activée."
        )
        return sauvegarder_csv(reponses)


# ============================================================
#                    SAUVEGARDE — CSV LOCAL (BACKUP)
# ============================================================

def sauvegarder_csv(reponses):
    """
    Sauvegarde de secours en CSV local.
    Utilisée quand Google Sheets n'est pas configuré.
    """

    fichier = "resultats_questionnaire.csv"

    # Vérifier si le fichier existe déjà
    fichier_existe = os.path.exists(fichier)

    with open(fichier, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reponses.keys())

        # Écrire les en-têtes si c'est un nouveau fichier
        if not fichier_existe:
            writer.writeheader()

        # Écrire la ligne de données
        writer.writerow(reponses)

    return True