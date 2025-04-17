import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import du chargeur de données
from utils.data_loader import DNSDataLoader

# Initialisation des états de session pour le défilement
if 'scroll_to' not in st.session_state:
    st.session_state.scroll_to = None


def show():
    st.title("🤖 Modélisation des Anomalies")

    # Initialisation du chargeur de données
    data_loader = DNSDataLoader()

    # Fonction pour préparer les données DNS
    def prepare_dns_data(df):
        # Création des features pour DNS
        df_agg = df.copy()
        df_agg["ratio_test_clients"] = df_agg["nb_test_dns"] / df_agg["nb_client_total"]
        df_agg["std_over_avg_dns"] = df_agg["std_dns_time"] / df_agg["avg_dns_time"]
        df_agg["avg_dns_per_client"] = df_agg["avg_dns_time"] / df_agg["nb_client_total"]
        df_agg["avg_dns_per_test_dns"] = df_agg["avg_dns_time"] / df_agg["nb_test_dns"]

        # Sélection des features pour le modèle
        features = [
            "avg_dns_time", "std_dns_time", "ratio_test_clients",
            "std_over_avg_dns", "avg_dns_per_client", "avg_dns_per_test_dns",
            "nb_test_dns", "nb_client_total"
        ]

        # Nettoyage des valeurs infinies ou NaN
        df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
        df_agg = df_agg.fillna(df_agg.mean(numeric_only=True))

        return df_agg, features

    # Fonction pour préparer les données Scoring
    def prepare_scoring_data(df):
        # Création des features pour Scoring
        df_agg = df.copy()
        df_agg["ratio_test_clients"] = df_agg["nb_test_scoring"] / df_agg["nb_client_total"]
        df_agg["score_per_test"] = df_agg["avg_score_scoring"] / df_agg["nb_test_scoring"]
        df_agg["latence_per_test"] = df_agg["avg_latence_scoring"] / df_agg["nb_test_scoring"]

        # Sélection des features pour le modèle
        features = [
            "avg_score_scoring", "ratio_test_clients", "latence_per_test",
            "score_per_test", "nb_test_scoring", "nb_client_total"
        ]

        # Nettoyage des valeurs infinies ou NaN
        df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
        df_agg = df_agg.fillna(df_agg.mean(numeric_only=True))

        return df_agg, features

    # Fonction pour application du modèle Isolation Forest
    def apply_isolation_forest(df, features, contamination):
        X = df[features]

        # Gestion des valeurs NaN
        X = X.fillna(X.mean())

        # Application du modèle
        clf = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly'] = clf.fit_predict(X)
        df['anomaly_score'] = clf.decision_function(X)

        # Conversion des prédictions (-1: anomalie, 1: normal) à (1: anomalie, 0: normal)
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

        return df

    # Fonction pour afficher les statistiques sur les anomalies
    def display_anomaly_stats(df, title, model_type):
        anomalies = df[df['anomaly'] == 1]

        st.subheader(f"Statistiques des anomalies - {title}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre total d'anomalies", len(anomalies))
        with col2:
            # Calcul du nombre de jours
            unique_dates = df['date_hour'].dt.date.unique()
            nb_days = len(unique_dates)
            avg_anomalies_per_day = len(anomalies) / nb_days if nb_days > 0 else 0
            st.metric("Anomalies moyennes par jour", f"{avg_anomalies_per_day:.2f}")
        with col3:
            anomaly_rate = len(anomalies) / len(df) * 100
            st.metric("Taux d'anomalies", f"{anomaly_rate:.2f}%")

        # Distribution temporelle des anomalies
        anomalies['date'] = anomalies['date_hour'].dt.date
        anomalies_by_date = anomalies.groupby('date').size().reset_index(name='count')

        if not anomalies_by_date.empty:
            fig = px.line(anomalies_by_date, x='date', y='count',
                          title=f"Distribution temporelle des anomalies - {title}",
                          labels={'count': "Nombre d'anomalies", 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)

        # Top des caractéristiques anomaliques
        if not anomalies.empty:
            st.subheader("Caractéristiques des anomalies")
            # Récupération de l'identifiant principal
            id_col = 'peag_nro' if 'peag_nro' in anomalies.columns else 'olt_name'

            # Top des entités avec le plus d'anomalies
            top_entities = anomalies.groupby(id_col).size().sort_values(ascending=False).reset_index(name='count')

            if not top_entities.empty:
                st.write(f"Top des {id_col} avec le plus d'anomalies:")

                # Stocker les résultats complets dans la session
                st.session_state[f"{model_type}_results"] = df
                st.session_state[f"{model_type}_id_col"] = id_col

                # Affichage du tableau des top 10
                st.dataframe(
                    top_entities.head(10),
                    use_container_width=True,
                    column_config={
                        "count": st.column_config.NumberColumn(
                            "Nombre d'anomalies",
                            help="Nombre total d'anomalies détectées"
                        )
                    },
                    hide_index=True
                )

                # Initialiser la clé de l'entité sélectionnée si elle n'existe pas
                selected_entity_key = f"{model_type}_selected_entity"
                if selected_entity_key not in st.session_state:
                    st.session_state[selected_entity_key] = top_entities[id_col].iloc[
                        0] if not top_entities.empty else None

                # Utiliser st.container pour maintenir la position du widget
                select_container = st.container()
                with select_container:
                    # Création d'un ancre pour le défilement
                    st.markdown(f"<div id='{model_type}_select_anchor'></div>", unsafe_allow_html=True)

                    # Créer un widget de sélection pour choisir une entité
                    if st.session_state[selected_entity_key] in top_entities[id_col].values:
                        default_index = top_entities[id_col].tolist().index(st.session_state[selected_entity_key])
                    else:
                        default_index = 0

                    selected_entity = st.selectbox(
                        f"Sélectionnez un {id_col} pour voir les détails:",
                        options=top_entities[id_col].tolist(),
                        index=default_index,
                        key=f"{model_type}_entity_selector"
                    )

                    # Mettre à jour l'état sélectionné
                    st.session_state[selected_entity_key] = selected_entity

                # Afficher les détails si une entité est sélectionnée
                if selected_entity:
                    # Créer une ancre avec l'ID unique pour cette entité
                    anchor_id = f"{model_type}_{selected_entity}"
                    st.session_state.scroll_to = anchor_id
                    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)

                    entity_data = df[df[id_col] == selected_entity]

                    st.markdown(f"### Détails pour {id_col}: **{selected_entity}**")

                    # Compteur d'anomalies pour cette entité
                    anomaly_count = entity_data['anomaly'].sum()
                    total_count = len(entity_data)
                    anomaly_rate = (anomaly_count / total_count * 100) if total_count > 0 else 0

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nombre d'anomalies", anomaly_count)
                    with col2:
                        st.metric("Taux d'anomalies", f"{anomaly_rate:.2f}%")

                    # Graphique avec la série temporelle pour cette entité
                    y_metric = 'avg_dns_time' if model_type == 'dns' else 'avg_score_scoring'
                    title_metric = 'Temps DNS moyen' if model_type == 'dns' else 'Score moyen'

                    fig = px.scatter(entity_data, x='date_hour', y=y_metric,
                                     color='anomaly', color_discrete_map={0: 'blue', 1: 'red'},
                                     title=f'{title_metric} pour {selected_entity}')

                    fig.update_layout(legend_title_text='Anomalie',
                                      xaxis_title="Date",
                                      yaxis_title=title_metric)

                    st.plotly_chart(fig, use_container_width=True)

                    # Statistiques descriptives
                    st.markdown("#### Statistiques descriptives")
                    st.dataframe(entity_data.describe().T)

    # Chargement des données brutes
    df_raw = data_loader.load_data()

    # Interface utilisateur
    st.subheader("Paramètres de modélisation")

    # Sélection de la période
    min_date = df_raw['date_hour'].min().date()
    max_date = df_raw['date_hour'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début",
                                   value=min_date,
                                   min_value=min_date,
                                   max_value=max_date)
    with col2:
        end_date = st.date_input("Date de fin",
                                 value=max_date,
                                 min_value=min_date,
                                 max_value=max_date)

    # Sélection du seuil de contamination
    contamination = st.slider("Seuil de contamination (% attendu d'anomalies)",
                              min_value=0.005, max_value=0.05, value=0.03, step=0.005,
                              format="%.3f",
                              help="Plus la valeur est élevée, plus le modèle détectera d'anomalies")

    # Chargement des données agrégées selon le niveau choisi
    # On charge les données après la sélection pour appliquer directement le filtre de date
    @st.cache_data
    def load_filtered_data(aggregation_level, start_date, end_date):
        df = data_loader.load_aggregated_data(aggregation_level)
        df_filtered = df[(df['date_hour'].dt.date >= start_date) &
                         (df['date_hour'].dt.date <= end_date)]
        return df_filtered

    # Création des onglets
    tabs = st.tabs(["🌐 Modèle DNS", "📊 Modèle Scoring", "🔄 Comparaison"])

    with tabs[0]:  # Tab DNS
        st.header("Détection d'anomalies sur les données DNS")

        df_dns = load_filtered_data(st.session_state.aggregation_level, start_date, end_date)

        # Vérifier si des données DNS existent
        if 'nb_test_dns' not in df_dns.columns or df_dns['nb_test_dns'].sum() == 0:
            st.warning("Pas de données DNS disponibles pour la période et le niveau d'agrégation sélectionnés.")
        else:
            df_dns = df_dns[df_dns['nb_test_dns'] > 0]

            # Préparation des données
            df_dns_prepared, features_dns = prepare_dns_data(df_dns)

            st.write("**Features utilisées pour le modèle DNS :**")
            st.write(", ".join(features_dns))

            # Application du modèle
            model_container = st.container()
            with model_container:
                if st.button("🚀 Lancer la détection d'anomalies DNS", key="dns_button"):
                    with st.spinner("Détection des anomalies en cours..."):
                        df_dns_results = apply_isolation_forest(df_dns_prepared, features_dns, contamination)

                        # Affichage des résultats
                        fig = px.scatter(df_dns_results, x='date_hour', y='avg_dns_time',
                                         color='anomaly', color_discrete_map={0: 'blue', 1: 'red'},
                                         hover_data=['peag_nro' if 'peag_nro' in df_dns_results.columns else None,
                                                     'olt_name' if 'olt_name' in df_dns_results.columns else None,
                                                     'anomaly_score'],
                                         title='Détection des anomalies DNS - Temps DNS moyen')

                        fig.update_layout(legend_title_text='Anomalie',
                                          xaxis_title="Date",
                                          yaxis_title="Temps DNS moyen (ms)")

                        st.plotly_chart(fig, use_container_width=True)

                        # Statistiques des anomalies
                        display_anomaly_stats(df_dns_results, "DNS", "dns")

                        # Sauvegarde des résultats dans la session
                        st.session_state.dns_results = df_dns_results

    with tabs[1]:  # Tab Scoring
        st.header("Détection d'anomalies sur les données de Scoring")

        df_scoring = load_filtered_data(st.session_state.aggregation_level, start_date, end_date)

        # Vérifier si des données Scoring existent
        if 'nb_test_scoring' not in df_scoring.columns or df_scoring['nb_test_scoring'].sum() == 0:
            st.warning("Pas de données Scoring disponibles pour la période et le niveau d'agrégation sélectionnés.")
        else:
            df_scoring = df_scoring[df_scoring['nb_test_scoring'] > 0]

            # Préparation des données
            df_scoring_prepared, features_scoring = prepare_scoring_data(df_scoring)

            st.write("**Features utilisées pour le modèle Scoring :**")
            st.write(", ".join(features_scoring))

            # Application du modèle
            model_container = st.container()
            with model_container:
                if st.button("🚀 Lancer la détection d'anomalies Scoring", key="scoring_button"):
                    with st.spinner("Détection des anomalies en cours..."):
                        df_scoring_results = apply_isolation_forest(df_scoring_prepared, features_scoring,
                                                                    contamination)

                        # Affichage des résultats
                        fig = px.scatter(df_scoring_results, x='date_hour', y='avg_score_scoring',
                                         color='anomaly', color_discrete_map={0: 'blue', 1: 'red'},
                                         hover_data=['peag_nro' if 'peag_nro' in df_scoring_results.columns else None,
                                                     'olt_name' if 'olt_name' in df_scoring_results.columns else None,
                                                     'anomaly_score'],
                                         title='Détection des anomalies Scoring - Score moyen')

                        fig.update_layout(legend_title_text='Anomalie',
                                          xaxis_title="Date",
                                          yaxis_title="Score moyen")

                        st.plotly_chart(fig, use_container_width=True)

                        # Statistiques des anomalies
                        display_anomaly_stats(df_scoring_results, "Scoring", "scoring")

                        # Sauvegarde des résultats dans la session
                        st.session_state.scoring_results = df_scoring_results

    with tabs[2]:  # Tab Comparaison
        st.header("Comparaison des anomalies DNS et Scoring")

        if 'dns_results' not in st.session_state or 'scoring_results' not in st.session_state:
            st.info("Veuillez d'abord lancer les modèles DNS et Scoring pour voir la comparaison.")
        else:
            # Fusionner les résultats sur la base des colonnes communes
            merge_cols = ['date_hour']
            if st.session_state.aggregation_level == "peag_nro":
                merge_cols.append('peag_nro')
            elif st.session_state.aggregation_level == "olt_name":
                merge_cols.append('olt_name')
            elif st.session_state.aggregation_level == "peag_nro & olt_name":
                merge_cols.extend(['peag_nro', 'olt_name'])

            # Sélectionner uniquement les colonnes nécessaires pour réduire la taille des données
            df_dns = st.session_state.dns_results[merge_cols + ['anomaly']].copy()
            df_scoring = st.session_state.scoring_results[merge_cols + ['anomaly']].copy()

            # Renommer les colonnes d'anomalies
            df_dns = df_dns.rename(columns={'anomaly': 'dns_anomaly'})
            df_scoring = df_scoring.rename(columns={'anomaly': 'scoring_anomaly'})

            # Fusionner les données
            df_combined = pd.merge(df_dns, df_scoring, on=merge_cols)

            # Calculer les statistiques de base
            normal_count = len(df_combined[(df_combined['dns_anomaly'] == 0) & (df_combined['scoring_anomaly'] == 0)])
            dns_only = len(df_combined[(df_combined['dns_anomaly'] == 1) & (df_combined['scoring_anomaly'] == 0)])
            scoring_only = len(df_combined[(df_combined['dns_anomaly'] == 0) & (df_combined['scoring_anomaly'] == 1)])
            both = len(df_combined[(df_combined['dns_anomaly'] == 1) & (df_combined['scoring_anomaly'] == 1)])
            total = len(df_combined)

            # Afficher les statistiques dans des colonnes
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points normaux", normal_count)
            with col2:
                st.metric("Anomalies DNS uniquement", dns_only)
            with col3:
                st.metric("Anomalies Scoring uniquement", scoring_only)
            with col4:
                st.metric("Doubles anomalies", both)

            # Calculer le taux de concordance
            concordance = (normal_count + both) / total * 100
            st.metric("Taux de concordance DNS/Scoring", f"{concordance:.2f}%")

            # Afficher le top 10 des entités avec doubles anomalies
            if both > 0:
                st.subheader("Top 10 des entités avec doubles anomalies")

                # Déterminer la colonne d'identifiant
                id_col = 'peag_nro' if 'peag_nro' in df_combined.columns else 'olt_name'

                # Filtrer les doubles anomalies et compter par entité
                double_anomalies = df_combined[(df_combined['dns_anomaly'] == 1) &
                                               (df_combined['scoring_anomaly'] == 1)]

                if not double_anomalies.empty and id_col in double_anomalies.columns:
                    top_double = double_anomalies.groupby(id_col).size().sort_values(ascending=False).head(10)
                    st.dataframe(top_double.reset_index(name='Nombre de doubles anomalies'))
                else:
                    st.warning("Aucune double anomalie trouvée ou colonne d'identifiant manquante.")

            # Préparer les données pour l'export
            st.subheader("Export des anomalies")

            # Créer un dataframe avec toutes les anomalies (DNS, Scoring ou les deux)
            anomalies_df = df_combined[(df_combined['dns_anomaly'] == 1) |
                                       (df_combined['scoring_anomaly'] == 1)].copy()

            # Ajouter une colonne pour le type d'anomalie
            anomalies_df['type_anomalie'] = 'DNS'
            anomalies_df.loc[anomalies_df['scoring_anomaly'] == 1, 'type_anomalie'] = 'Scoring'
            anomalies_df.loc[(anomalies_df['dns_anomaly'] == 1) &
                             (anomalies_df['scoring_anomaly'] == 1), 'type_anomalie'] = 'Double'

            # Formater la date pour l'export
            anomalies_df['date'] = anomalies_df['date_hour'].dt.date
            anomalies_df['heure'] = anomalies_df['date_hour'].dt.time

            # Sélectionner les colonnes à exporter
            export_cols = ['date', 'heure', 'type_anomalie']
            if 'peag_nro' in anomalies_df.columns:
                export_cols.append('peag_nro')
            if 'olt_name' in anomalies_df.columns:
                export_cols.append('olt_name')

            # Afficher un aperçu des données
            st.dataframe(anomalies_df[export_cols].head(100))  # Limité à 100 lignes pour l'aperçu

            # Option de téléchargement
            csv = anomalies_df[export_cols].to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les anomalies au format CSV",
                data=csv,
                file_name=f"anomalies_{start_date}_{end_date}.csv",
                mime="text/csv",
                help="Export de toutes les anomalies détectées (DNS, Scoring et doubles)"
            )

    # Ajout du script JavaScript pour scroller à la position sauvegardée
    if st.session_state.scroll_to:
        js = f"""
        <script>
            function scroll_to(id) {{
                var element = document.getElementById(id);
                if (element) {{
                    element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}
            // Utiliser setTimeout pour s'assurer que le DOM est complètement chargé
            setTimeout(function() {{
                scroll_to("{st.session_state.scroll_to}");
            }}, 100);
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)