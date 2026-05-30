import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import csv
import os

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, "google_credentials.json")
SHEET_NAME = "Resultats_PFE_TGR"

LIKERT_OPTIONS = {
    1: "Pas du tout d'accord",
    2: "Pas d'accord",
    3: "Neutre",
    4: "D'accord",
    5: "Tout à fait d'accord"
}

# ============================================================
# PROFIL SOCIODÉMOGRAPHIQUE
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
# ITEMS LIKERT
# ============================================================

QUESTIONNAIRE_SECTIONS = {
    "qualite_percue": {
        "titre": " Qualité perçue des réponses du chatbot",
        "consigne": "Évaluez la qualité des réponses que vous avez reçues du chatbot.",
        "items": {
            "QP1": "Les réponses fournies par le chatbot étaient pertinentes par rapport à mes questions.",
            "QP2": "Les informations fournies me semblaient correctes et fiables.",
            "QP3": "Les réponses étaient suffisamment détaillées et complètes.",
            "QP4": "Le langage utilisé par le chatbot était clair et compréhensible.",
            "QP5": "Le chatbot a répondu rapidement à mes questions."
        }
    },
    "satisfaction": {
        "titre": " Satisfaction globale",
        "consigne": "Évaluez votre niveau de satisfaction suite à votre interaction.",
        "items": {
            "SAT1": "Je suis globalement satisfait(e) de mon interaction avec le chatbot.",
            "SAT2": "L'expérience avec le chatbot a répondu à mes attentes.",
            "SAT3": "L'interaction avec le chatbot a été une expérience positive."
        }
    },
    "confiance": {
        "titre": " Confiance institutionnelle envers la TGR",
        "consigne": "Évaluez votre perception de la TGR après cette interaction.",
        "items": {
            "CI1": "Après cette interaction, je perçois la TGR comme une institution compétente.",
            "CI2": "Je fais confiance à la TGR pour fournir des informations fiables et exactes.",
            "CI3": "J'ai le sentiment que la TGR se soucie véritablement des besoins de ses usagers."
        }
    },
    "utilite_percue": {
        "titre": " Utilité perçue de l'IA",
        "consigne": "Évaluez votre perception de l'utilité de l'IA dans le service public.",
        "items": {
            "UP1": "L'utilisation d'un chatbot IA est utile pour obtenir des informations sur les services publics.",
            "UP2": "Un chatbot IA peut me permettre de gagner du temps dans mes démarches administratives.",
            "UP3": "Un chatbot IA améliore mon accès à l'information par rapport aux canaux traditionnels."
        }
    },
    "facilite_percue": {
        "titre": " Facilité d'utilisation",
        "consigne": "Évaluez la facilité d'utilisation du chatbot.",
        "items": {
            "FU1": "L'interaction avec le chatbot était facile et intuitive.",
            "FU2": "Je n'ai pas eu besoin d'effort particulier pour utiliser le chatbot."
        }
    }
}

# ============================================================
# FONCTIONS D'AFFICHAGE
# ============================================================

def afficher_profil():
    """Affiche et collecte les questions sociodémographiques."""
    st.markdown("### Votre profil")
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
    st.markdown("##  Donnez-nous votre avis !")

    st.markdown("""
    Merci d'avoir testé le chatbot ! 
    
    Répondez aux questions suivantes pour nous aider à évaluer 
    cette expérience. **Il n'y a pas de bonne ou de mauvaise réponse.**
    Seule votre perception compte.
    
     Durée : **4-5 minutes**
    """)

    st.markdown("---")

    toutes_reponses = {}

    profil = afficher_profil()
    toutes_reponses.update(profil)

    st.markdown("---")

    for _, section in QUESTIONNAIRE_SECTIONS.items():
        st.markdown(f"### {section['titre']}")
        st.markdown(f"*{section['consigne']}*")
        st.markdown("---")

        for code, texte in section["items"].items():
            reponse = afficher_likert(code, texte)
            toutes_reponses[code] = reponse

    st.markdown("###  Commentaire libre (facultatif)")
    commentaire = st.text_area(
        "Si vous souhaitez partager une remarque sur votre expérience :",
        key="commentaire_libre",
        height=100,
        placeholder="Votre commentaire ici..."
    )
    toutes_reponses["commentaire"] = commentaire

    return toutes_reponses


# ============================================================
# VÉRIFICATION DES RÉPONSES
# ============================================================

def verifier_reponses(reponses):
    """Vérifie que toutes les réponses obligatoires sont remplies."""
    for key in PROFIL_QUESTIONS.keys():
        if reponses.get(key) == "— Sélectionnez —":
            return False, f"Veuillez remplir le champ : {PROFIL_QUESTIONS[key]['label']}"
    return True, ""


# ============================================================
# CONNEXION GOOGLE SHEETS
# ============================================================

def get_google_sheet():
    """
    Retourne la première feuille du Google Sheets.
    Priorité : Streamlit Secrets → fichier local.
    """
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(
                creds_dict,
                scopes=scope
            )
            client = gspread.authorize(creds)
            return client.open(SHEET_NAME).sheet1
    except Exception as e:
        st.error(f"Erreur Streamlit Secrets : {type(e).__name__} - {e}")

    if os.path.exists(CREDENTIALS_FILE):
        try:
            creds = Credentials.from_service_account_file(
                CREDENTIALS_FILE,
                scopes=scope
            )
            client = gspread.authorize(creds)
            return client.open(SHEET_NAME).sheet1
        except Exception as e:
            raise Exception(f"Erreur Google Sheets/local : {type(e).__name__} - {e}")

    raise FileNotFoundError(f"Fichier credentials introuvable : {CREDENTIALS_FILE}")


# ============================================================
# SAUVEGARDE — GOOGLE SHEETS
# ============================================================

def sauvegarder_google_sheets(reponses):
    """
    Sauvegarde les réponses dans Google Sheets.
    Si Google Sheets échoue, bascule sur CSV local.
    """
    donnees = reponses.copy()
    donnees["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    donnees["nb_echanges"] = st.session_state.get("nb_messages", 0)

    try:
        sheet = get_google_sheet()

        premiere_ligne = sheet.row_values(1)

        if not premiere_ligne:
            sheet.insert_row(list(donnees.keys()), index=1)
            prochaine_ligne = 2
        else:
            prochaine_ligne = len(sheet.col_values(1)) + 1

        sheet.insert_row(list(donnees.values()), index=prochaine_ligne)

        st.success(f" Réponses enregistrées dans Google Sheets (ligne {prochaine_ligne}).")
        return True

    except Exception as e:
        st.warning(
            f" Google Sheets non disponible ({type(e).__name__} - {e}). "
            f"Sauvegarde locale en CSV activée."
        )
        return sauvegarder_csv(donnees)


# ============================================================
# SAUVEGARDE — CSV LOCAL (BACKUP)
# ============================================================

def sauvegarder_csv(reponses):
    """Sauvegarde de secours en CSV local."""
    fichier = os.path.join(BASE_DIR, "resultats_questionnaire.csv")
    fichier_exist = os.path.exists(fichier)

    with open(fichier, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reponses.keys())

        if not fichier_exist:
            writer.writeheader()

        writer.writerow(reponses)

    st.success(" Réponses enregistrées en local dans resultats_questionnaire.csv")
    return True
