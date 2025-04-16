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
                st.write(f"Top des {id_col} avec le plus d'anomalies (cliquez sur une ligne pour plus de détails):")
                
                # Stocker les données pour l'interaction
                st.session_state[f"{model_type}_top_entities"] = top_entities
                st.session_state[f"{model_type}_anomalies"] = anomalies
                st.session_state[f"{model_type}_all_data"] = df
                st.session_state[f"{model_type}_id_col"] = id_col
                
                # Tableau interactif avec gestion des clics
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

                if not top_entities.empty:
                    id_col = st.session_state[f"{model_type}_id_col"]
                    selected_entity = st.selectbox(
                        f"Sélectionnez un {id_col} pour voir les détails:",
                        options=top_entities[id_col].tolist(),
                        key=f"{model_type}_entity_selector"
                    )
                    st.session_state[f"{model_type}_selected_entity"] = selected_entity
                
                # Afficher le graphique pour l'entité sélectionnée (si disponible)
                selected_entity = st.session_state.get(f"{model_type}_selected_entity", None)
                
                if selected_entity:
                    display_entity_details(
                        df=df, 
                        entity_name=selected_entity, 
                        id_col=id_col, 
                        model_type=model_type
                    )
    
    # Fonction pour afficher les détails d'une entité spécifique
    def display_entity_details(df, entity_name, id_col, model_type):
        entity_data = df[df[id_col] == entity_name]
        
        st.markdown(f"### Détails pour {id_col}: **{entity_name}**")
        
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
                         title=f'{title_metric} pour {entity_name} (rouge = anomalie)')
        
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
                              min_value=0.01, max_value=0.3, value=0.05, step=0.01,
                              format="%.2f",
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
            if st.button("🚀 Lancer la détection d'anomalies Scoring", key="scoring_button"):
                with st.spinner("Détection des anomalies en cours..."):
                    df_scoring_results = apply_isolation_forest(df_scoring_prepared, features_scoring, contamination)
                    
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
            
            df_combined = pd.merge(
                st.session_state.dns_results[merge_cols + ['anomaly']].rename(columns={'anomaly': 'dns_anomaly'}),
                st.session_state.scoring_results[merge_cols + ['anomaly']].rename(columns={'anomaly': 'scoring_anomaly'}),
                on=merge_cols
            )
            
            # Créer une catégorie combinée
            df_combined['combined'] = 'Normal'
            df_combined.loc[df_combined['dns_anomaly'] == 1, 'combined'] = 'DNS Anomalie'
            df_combined.loc[df_combined['scoring_anomaly'] == 1, 'combined'] = 'Scoring Anomalie'
            df_combined.loc[(df_combined['dns_anomaly'] == 1) & (df_combined['scoring_anomaly'] == 1), 'combined'] = 'Double Anomalie'
            
            # Visualisation
            fig = px.scatter(df_combined, x='date_hour', y=range(len(df_combined)), 
                             color='combined', 
                             color_discrete_map={
                                 'Normal': 'blue',
                                 'DNS Anomalie': 'orange',
                                 'Scoring Anomalie': 'green',
                                 'Double Anomalie': 'red'
                             },
                             hover_data=merge_cols,
                             title='Comparaison des anomalies DNS et Scoring')
            
            fig.update_layout(legend_title_text='Type',
                             xaxis_title="Date",
                             yaxis_title="Index")
            fig.update_yaxes(showticklabels=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques de comparaison
            st.subheader("Statistiques de comparaison")
            
            # Metrics dans des colonnes
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                normal_count = len(df_combined[df_combined['combined'] == 'Normal'])
                st.metric("Points normaux", normal_count)
            with col2:
                dns_only = len(df_combined[df_combined['combined'] == 'DNS Anomalie'])
                st.metric("Anomalies DNS uniquement", dns_only)
            with col3:
                scoring_only = len(df_combined[df_combined['combined'] == 'Scoring Anomalie'])
                st.metric("Anomalies Scoring uniquement", scoring_only)
            with col4:
                both = len(df_combined[df_combined['combined'] == 'Double Anomalie'])
                st.metric("Doubles anomalies", both)
            
            # Matrice de confusion
            confusion_matrix = pd.crosstab(df_combined['dns_anomaly'], df_combined['scoring_anomaly'], 
                                           rownames=['DNS'], colnames=['Scoring'])
            
            st.subheader("Matrice de concordance des anomalies")
            st.dataframe(confusion_matrix)
            
            # Taux de concordance
            total = len(df_combined)
            concordance = (normal_count + both) / total * 100
            st.metric("Taux de concordance DNS/Scoring", f"{concordance:.2f}%")
            
            # Liste des anomalies
            st.subheader("Liste des anomalies détectées")
            anomalies_df = df_combined[df_combined['combined'] != 'Normal'].copy()
            anomalies_df['date'] = anomalies_df['date_hour'].dt.date
            anomalies_df['hour'] = anomalies_df['date_hour'].dt.hour
            st.dataframe(anomalies_df)
            
            # Option de téléchargement
            csv = anomalies_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les résultats des anomalies",
                data=csv,
                file_name=f"anomalies_{start_date}_{end_date}.csv",
                mime="text/csv",
            )
            
            # Top des entités avec des doubles anomalies
            if both > 0:
                st.subheader("Entités avec des doubles anomalies")
                id_col = merge_cols[1] if len(merge_cols) > 1 else None
                if id_col:
                    double_anomalies = df_combined[df_combined['combined'] == 'Double Anomalie']
                    top_double = double_anomalies.groupby(id_col).size().sort_values(ascending=False).reset_index(name='count')
                    st.dataframe(top_double.head(10), use_container_width=True)