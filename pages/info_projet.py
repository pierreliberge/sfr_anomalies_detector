import streamlit as st

def show():
    st.title("ℹ️ Informations sur le projet")
    st.markdown("""
    Ce projet a été réalisé dans le cadre du **Master 2 MOSEF**, en partenariat avec :

    - 📡 **SFR**
    - 📊 Le cabinet de conseil **Nexialog**
    - 🎓 L'Université **Paris 1 Panthéon-Sorbonne**

    L’objectif est de fournir une solution permettant aux équipes métiers de :
    - détecter des anomalies sur le réseau,
    - comprendre les causes probables,
    - et à terme, **prévoir ces anomalies** pour anticiper les incidents réseau.

    ---
    ### ⚙️ Données utilisées dans l’application

    Pour garantir un fonctionnement fluide de l'application, nous avons **échantillonné** le dataset d'origine :
    
    - 📅 **Période retenue** : du **1er décembre** au **1er janvier**
    - 📉 Ce sous-ensemble permet de réduire la charge mémoire et le temps de traitement, tout en restant représentatif des anomalies typiques du réseau.
    
    Cette période a été utilisée pour l’analyse exploratoire (_EDA_) ainsi que pour la modélisation.
    """)
