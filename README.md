# SFR Anomalies Detector

Application Streamlit pour analyser et détecter des anomalies réseaux SFR. Les données étant confidentielles, merci de les importer vous même dans le dossier cloné.

## Installation

1. **Cloner le dépôt** :

git clone https://github.com/pierreliberge/sfr_anomalies_detector.git

cd sfr_anomalies_detector

2. **Ajouter le fichier de données** :

Copier 250327_tests_fixe_dns_sah_202412_202501.parquet à la racine du projet :

sfr_anomalies_detector/250327_tests_fixe_dns_sah_202412_202501.parquet

3. **Lancer le script de préparation** :

python data_sampler.py

4. **Lancer l’application** :

streamlit run app.py
