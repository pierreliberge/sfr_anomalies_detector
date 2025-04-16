import streamlit as st
from pages import eda, modelisation, info_projet

# Configuration de l'application
st.set_page_config(page_title="SFR anomalies detector", layout="wide")

# Import CSS personnalisé
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialisation des états
if 'aggregation_level' not in st.session_state or st.session_state.aggregation_level is None:
    st.session_state.aggregation_level = "peag_nro"

if 'subpage' not in st.session_state:
    st.session_state.subpage = "home"

if 'eda_view' not in st.session_state:
    st.session_state.eda_view = None

# === SIDEBAR ===
st.sidebar.title("🧭 Navigation")
subpage_choice = st.sidebar.radio(
    "Aller vers :",
    ["🏠 Accueil", "🔍 EDA", "🤖 Modélisation", "ℹ️ Infos projet"],
    index=["home", "eda", "modelisation", "info"].index(st.session_state.subpage),
    format_func=lambda x: x.replace("🏠 ", "").replace("🔍 ", "").replace("🤖 ", "").replace("ℹ️ ", "")
)

subpage_mapping = {
    "🏠 Accueil": "home",
    "🔍 EDA": "eda",
    "🤖 Modélisation": "modelisation",
    "ℹ️ Infos projet": "info"
}

st.session_state.subpage = subpage_mapping[subpage_choice]

# === HEADER PRINCIPAL ===
st.markdown("""
    <div class="main-header">
        <div class="title-text">SFR anomalies detector</div>
    </div>
    <hr>
""", unsafe_allow_html=True)

st.markdown(f"### 🔧 Agrégation en cours : **{st.session_state.aggregation_level}**")

# === CONTENU PRINCIPAL ===
if st.session_state.subpage == "home":
    st.markdown("## 📶 Bienvenue sur **SFR anomalies detector**")
    st.markdown("#### Créé par Pierre Liberge")
    st.write("""
        Cette application a pour but d'explorer et de modéliser des anomalies sur le réseau fixe SFR.
        Merci de choisir ci-dessous le niveau d'agrégation des données que vous souhaitez analyser :
    """)

    current_level = st.session_state.aggregation_level or "peag_nro"
    aggregation_level = st.radio(
        "Niveau d'agrégation :",
        ["peag_nro", "olt_name", "peag_nro & olt_name"],
        index=["peag_nro", "olt_name", "peag_nro & olt_name"].index(current_level),
        key="agg_level_radio",
        horizontal=True
    )

    if st.button("✅ Valider le niveau d'agrégation"):
        st.session_state.aggregation_level = aggregation_level
        st.success(f"Niveau d'agrégation défini : {aggregation_level}")
        st.rerun()

elif st.session_state.subpage == "eda":
    eda_choice = st.radio("Choisissez une sous-analyse EDA :", ["EDA DNS", "EDA Scoring"], horizontal=True)
    if st.button("▶️ Lancer l’analyse EDA", type="primary"):
        st.session_state.eda_view = eda_choice
        st.rerun()

    if st.session_state.eda_view:
        eda.show(st.session_state.eda_view)

elif st.session_state.subpage == "modelisation":
    modelisation.show()

elif st.session_state.subpage == "info":
    info_projet.show()
