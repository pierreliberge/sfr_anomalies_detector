import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.data_loader import DNSDataLoader
from datetime import datetime, timedelta
import numpy as np

def show_error_no_aggregation():
    """Affiche un message d'erreur si aucun niveau d'agrégation n'est sélectionné"""
    st.error("⚠️ Aucun niveau d'agrégation sélectionné !")
    st.markdown("""
    Merci de retourner sur la page d'accueil pour sélectionner un niveau d'agrégation
    avant de pouvoir continuer l'analyse des données.
    
    [Retour à l'accueil](/?page=home)
    """)

class Visualizations:
    """Classe pour gérer les visualisations"""
    
    @staticmethod
    def time_series_chart(df, y_column):
        """Graphique de série temporelle (agrégé par heure)"""
        df_agg = df.groupby("date_hour", as_index=False)[y_column].mean()

        fig = px.line(
            df_agg, 
            x="date_hour", 
            y=y_column,
            title=f"Évolution moyenne de {y_column} au cours du temps"
        )
        fig.update_layout(
            xaxis_title="Date et heure",
            yaxis_title=y_column,
            template="plotly_white",
            height=400
        )
        return fig

    
    @staticmethod
    def distribution_chart(df, column):
        """Graphique de distribution"""
        fig = px.histogram(
            df, 
            x=column, 
            nbins=30,
            title=f"Distribution de {column}"
        )
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Fréquence",
            template="plotly_white",
            height=400
        )
        return fig
    
    @staticmethod
    def scatter_plot(df, x_column, y_column, color_column=None):
        """Graphique de dispersion"""
        fig = px.scatter(
            df, 
            x=x_column, 
            y=y_column,
            color=color_column,
            title=f"Relation entre {x_column} et {y_column}"
        )
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white",
            height=400
        )
        return fig
    
    @staticmethod
    def aggregate_bar_chart(df, group_col, value_col):
        """Graphique agrégé en barres"""
        agg_df = df.groupby(group_col)[value_col].mean().reset_index()
        agg_df = agg_df.sort_values(value_col, ascending=False).head(15)
        
        fig = px.bar(
            agg_df, 
            x=group_col, 
            y=value_col,
            title=f"Moyenne de {value_col} par {group_col}"
        )
        fig.update_layout(
            xaxis_title=group_col,
            yaxis_title=f"Moyenne de {value_col}",
            template="plotly_white",
            height=400
        )
        return fig
    
    @staticmethod
    def heatmap(df, x_col, y_col, value_col):
        """Carte de chaleur"""
        # Création d'un pivot pour la carte de chaleur
        pivot_df = df.pivot_table(
            index=y_col, 
            columns=x_col, 
            values=value_col, 
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdBu_r'
        ))
        
        fig.update_layout(
            title=f"Carte de chaleur : {value_col} par {x_col} et {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )
        return fig

def filter_data(df, filters):
    """Filtre le dataframe selon les filtres choisis"""
    filtered_df = df.copy()
    
    for col, value in filters.items():
        if value is not None and col in filtered_df.columns:
            if isinstance(value, tuple) and len(value) == 2:  # Pour les plages (sliders)
                if pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                    filtered_df = filtered_df[(filtered_df[col] >= value[0]) & (filtered_df[col] <= value[1])]
                else:
                    filtered_df = filtered_df[(filtered_df[col] >= value[0]) & (filtered_df[col] <= value[1])]
            elif isinstance(value, list):  # Pour les multiselect
                if value:  # Si la liste n'est pas vide
                    filtered_df = filtered_df[filtered_df[col].isin(value)]
            else:  # Pour les sélections simples
                filtered_df = filtered_df[filtered_df[col] == value]
    
    return filtered_df

def show(view_type):
    st.title("🔍 Analyse exploratoire (EDA)")

    if 'aggregation_level' not in st.session_state or not st.session_state.aggregation_level:
        show_error_no_aggregation()
        return

    view = view_type
    data_loader = DNSDataLoader()
    df = data_loader.load_aggregated_data(st.session_state.aggregation_level, view)

    st.markdown(f"### Niveau d'agrégation actuel : **{st.session_state.aggregation_level}**")
    st.markdown(f"### Type d'analyse : **{view}**")

    common_cols = ["date_hour", "code_departement", "olt_model", "olt_name", "peag_nro", "boucle", "dsp", "pebib", "nb_client_total"]

    if view == "EDA DNS":
        columns_to_keep = common_cols + ["pop_dns", "nb_test_dns", "avg_dns_time", "std_dns_time"]
    else:
        columns_to_keep = common_cols + ["nb_test_scoring", "avg_latence_scoring", "std_latence_scoring", "avg_score_scoring", "std_score_scoring"]

    df = df[[col for col in df.columns if col in columns_to_keep]]
    
    # Création des sections de filtres avec un effet d'expansion/réduction
    with st.expander("🔍 Filtres généraux", expanded=False):
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        filters = {}
        
        # Première colonne de filtres généraux
        with col1:
            # Filtre de date
            if 'date_hour' in df.columns:
                min_date = df['date_hour'].min().date()
                max_date = df['date_hour'].max().date()
                date_range = st.date_input(
                    "Période d'analyse",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filters['date_hour'] = (
                        pd.Timestamp(start_date), 
                        pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    )
            
            # Département
            if 'code_departement' in df.columns:
                unique_deps = sorted(df['code_departement'].dropna().unique())
                selected_deps = st.multiselect("Code département", options=unique_deps)
                if selected_deps:
                    filters['code_departement'] = selected_deps
        
        # Deuxième colonne de filtres généraux
        with col2:
            # Modèle OLT
            if 'olt_model' in df.columns:
                unique_models = sorted(df['olt_model'].dropna().unique())
                selected_models = st.multiselect("Modèle OLT", options=unique_models)
                if selected_models:
                    filters['olt_model'] = selected_models
            
            # Afficher les filtres spécifiques en fonction du niveau d'agrégation
            if st.session_state.aggregation_level in ["peag_nro", "peag_nro & olt_name"] and 'peag_nro' in df.columns:
                unique_peag = sorted(df['peag_nro'].dropna().unique())
                selected_peag = st.multiselect("PEAG/NRO", options=unique_peag)
                if selected_peag:
                    filters['peag_nro'] = selected_peag
                    
            if st.session_state.aggregation_level in ["olt_name", "peag_nro & olt_name"] and 'olt_name' in df.columns:
                unique_olt = sorted(df['olt_name'].dropna().unique())
                selected_olt = st.multiselect("Nom OLT", options=unique_olt)
                if selected_olt:
                    filters['olt_name'] = selected_olt
        
        # Autres filtres généraux (ligne complète)
        cols = st.columns(3)
        
        with cols[0]:
            if 'boucle' in df.columns:
                unique_boucle = sorted(df['boucle'].dropna().unique())
                selected_boucle = st.multiselect("Boucle", options=unique_boucle)
                if selected_boucle:
                    filters['boucle'] = selected_boucle
        
        with cols[1]:
            if 'dsp' in df.columns:
                unique_dsp = sorted(df['dsp'].dropna().unique())
                selected_dsp = st.multiselect("DSP", options=unique_dsp)
                if selected_dsp:
                    filters['dsp'] = selected_dsp
        
        with cols[2]:
            if 'pebib' in df.columns:
                unique_pebib = sorted(df['pebib'].dropna().unique())
                selected_pebib = st.multiselect("PEBIB", options=unique_pebib)
                if selected_pebib:
                    filters['pebib'] = selected_pebib
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filtres techniques dans un autre expander
    with st.expander("🛠️ Filtres techniques", expanded=False):
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Première colonne de filtres techniques
        with col1:
            # Nombre de clients total
            if 'nb_client_total' in df.columns:
                min_clients = int(df['nb_client_total'].min())
                max_clients = int(df['nb_client_total'].max())
                selected_clients = st.slider(
                    "Nombre de clients total", 
                    min_value=min_clients,
                    max_value=max_clients,
                    value=(min_clients, max_clients)
                )
                if selected_clients != (min_clients, max_clients):
                    filters['nb_client_total'] = selected_clients
            
            # Nombre de tests DNS
            if 'nb_test_dns' in df.columns:
                min_tests = int(df['nb_test_dns'].min())
                max_tests = int(df['nb_test_dns'].max())
                selected_tests = st.slider(
                    "Nombre de tests DNS", 
                    min_value=min_tests,
                    max_value=max_tests,
                    value=(min_tests, max_tests)
                )
                if selected_tests != (min_tests, max_tests):
                    filters['nb_test_dns'] = selected_tests
            
            # Temps DNS moyen
            if 'avg_dns_time' in df.columns:
                min_time = float(df['avg_dns_time'].min())
                max_time = float(df['avg_dns_time'].max())
                selected_time = st.slider(
                    "Temps DNS moyen (ms)", 
                    min_value=min_time,
                    max_value=max_time,
                    value=(min_time, max_time),
                    format="%.2f"
                )
                if selected_time != (min_time, max_time):
                    filters['avg_dns_time'] = selected_time
        
        # Deuxième colonne de filtres techniques
        with col2:
            # Écart-type temps DNS
            if 'std_dns_time' in df.columns:
                min_std = float(df['std_dns_time'].min())
                max_std = float(df['std_dns_time'].max())
                selected_std = st.slider(
                    "Écart-type temps DNS (ms)", 
                    min_value=min_std,
                    max_value=max_std,
                    value=(min_std, max_std),
                    format="%.2f"
                )
                if selected_std != (min_std, max_std):
                    filters['std_dns_time'] = selected_std
            
            # POP DNS si disponible
            if 'pop_dns' in df.columns:
                unique_pop = sorted(df['pop_dns'].dropna().unique())
                selected_pop = st.multiselect("POP DNS", options=unique_pop)
                if selected_pop:
                    filters['pop_dns'] = selected_pop
            
            # Nombre de tests scoring
            if 'nb_test_scoring' in df.columns:
                min_scoring = int(df['nb_test_scoring'].min())
                max_scoring = int(df['nb_test_scoring'].max())
                selected_scoring = st.slider(
                    "Nombre de tests scoring", 
                    min_value=min_scoring,
                    max_value=max_scoring,
                    value=(min_scoring, max_scoring)
                )
                if selected_scoring != (min_scoring, max_scoring):
                    filters['nb_test_scoring'] = selected_scoring
        
        # Troisième ligne de filtres techniques (s'il y a d'autres métriques)
        cols = st.columns(3)
        
        with cols[0]:
            if 'avg_latence_scoring' in df.columns:
                min_latence = float(df['avg_latence_scoring'].min())
                max_latence = float(df['avg_latence_scoring'].max())
                selected_latence = st.slider(
                    "Latence moyenne scoring (ms)", 
                    min_value=min_latence,
                    max_value=max_latence,
                    value=(min_latence, max_latence),
                    format="%.2f"
                )
                if selected_latence != (min_latence, max_latence):
                    filters['avg_latence_scoring'] = selected_latence
        
        with cols[1]:
            if 'std_latence_scoring' in df.columns:
                min_std_lat = float(df['std_latence_scoring'].min())
                max_std_lat = float(df['std_latence_scoring'].max())
                selected_std_lat = st.slider(
                    "Écart-type latence scoring (ms)", 
                    min_value=min_std_lat,
                    max_value=max_std_lat,
                    value=(min_std_lat, max_std_lat),
                    format="%.2f"
                )
                if selected_std_lat != (min_std_lat, max_std_lat):
                    filters['std_latence_scoring'] = selected_std_lat
        
        with cols[2]:
            if 'avg_score_scoring' in df.columns:
                min_score = float(df['avg_score_scoring'].min())
                max_score = float(df['avg_score_scoring'].max())
                selected_score = st.slider(
                    "Score moyen scoring", 
                    min_value=min_score,
                    max_value=max_score,
                    value=(min_score, max_score),
                    format="%.2f"
                )
                if selected_score != (min_score, max_score):
                    filters['avg_score_scoring'] = selected_score
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Appliquer les filtres aux données
    filtered_df = filter_data(df, filters)
    
    # Afficher un résumé des filtres
    active_filters = {k: v for k, v in filters.items() if v is not None}
    if active_filters:
        st.markdown("### Filtres actifs")
        filter_summary = []
        
        for col, value in active_filters.items():
            if isinstance(value, tuple) and len(value) == 2:
                if col == 'date_hour':
                    filter_summary.append(f"📅 **{col}**: du {value[0].strftime('%d/%m/%Y')} au {value[1].strftime('%d/%m/%Y')}")
                else:
                    filter_summary.append(f"📊 **{col}**: entre {value[0]} et {value[1]}")
            elif isinstance(value, list):
                filter_summary.append(f"📋 **{col}**: {', '.join(map(str, value))}")
            else:
                filter_summary.append(f"🔍 **{col}**: {value}")
        
        st.markdown(" | ".join(filter_summary))
    
    # Afficher le nombre de résultats après filtrage
    st.info(f"📊 {len(filtered_df)} résultats après filtrage sur un total de {len(df)} enregistrements")
    
    # Si aucune donnée après filtrage
    if filtered_df.empty:
        st.warning("Aucune donnée ne correspond aux critères de filtrage sélectionnés.")
        return
    
    # Section des visualisations
    st.markdown("## 📊 Visualisations")
    
    # Sélection des visualisations
    viz_options = st.selectbox(
        "Choisir un type de visualisation",
        [
            "Évolution temporelle",
            "Distribution des variables",
            "Relation entre variables",
            "Comparaison par catégorie",
            "Carte de chaleur"
        ]
    )
    
    # En fonction du choix de visualisation
    if viz_options == "Évolution temporelle":
        num_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col]) and col != 'date_hour']
        selected_metric = st.selectbox("Métrique à visualiser", num_cols)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Graphique d'évolution temporelle
            fig = Visualizations.time_series_chart(filtered_df, selected_metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Pas assez de données pour afficher ce graphique.")
        
        with col2:
            # Statistiques descriptives
            st.markdown("### Statistiques")
            stats_df = filtered_df[selected_metric].describe().reset_index()
            stats_df.columns = ['Statistique', 'Valeur']
            st.dataframe(stats_df, hide_index=True)
            
            # Tendance
            if len(filtered_df) > 1:
                first_val = filtered_df[selected_metric].iloc[0]
                last_val = filtered_df[selected_metric].iloc[-1]
                percent_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                
                st.markdown("### Tendance")
                if percent_change > 0:
                    st.markdown(f"📈 **Hausse de {percent_change:.2f}%**")
                elif percent_change < 0:
                    st.markdown(f"📉 **Baisse de {abs(percent_change):.2f}%**")
                else:
                    st.markdown("➡️ **Stable**")
    
    elif viz_options == "Distribution des variables":
        num_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
        selected_var = st.selectbox("Variable à analyser", num_cols)
        
        fig = Visualizations.distribution_chart(filtered_df, selected_var)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Statistiques")
            stats_df = filtered_df[selected_var].describe().reset_index()
            stats_df.columns = ['Statistique', 'Valeur']
            st.dataframe(stats_df, hide_index=True)
        
        with col2:
            st.markdown("### Distribution")
            
            # Calculer le pourcentage de valeurs dans différentes tranches
            q1 = filtered_df[selected_var].quantile(0.25)
            median = filtered_df[selected_var].quantile(0.5)
            q3 = filtered_df[selected_var].quantile(0.75)
            
            below_q1 = (filtered_df[selected_var] < q1).mean() * 100
            q1_to_median = ((filtered_df[selected_var] >= q1) & (filtered_df[selected_var] < median)).mean() * 100
            median_to_q3 = ((filtered_df[selected_var] >= median) & (filtered_df[selected_var] < q3)).mean() * 100
            above_q3 = (filtered_df[selected_var] >= q3).mean() * 100
            
            st.markdown(f"- 🔻 Bas (< Q1): **{below_q1:.1f}%**")
            st.markdown(f"- ⬇️ Moyen-bas (Q1-médiane): **{q1_to_median:.1f}%**")
            st.markdown(f"- ⬆️ Moyen-haut (médiane-Q3): **{median_to_q3:.1f}%**")
            st.markdown(f"- 🔺 Haut (> Q3): **{above_q3:.1f}%**")
            
            # Valeurs aberrantes
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((filtered_df[selected_var] < lower_bound) | (filtered_df[selected_var] > upper_bound)).mean() * 100
            
            st.markdown(f"- ⚠️ Valeurs aberrantes: **{outliers:.1f}%**")
    
    elif viz_options == "Relation entre variables":
        num_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Variable X", num_cols, index=0)
        
        with col2:
            available_y_vars = [col for col in num_cols if col != x_var]
            y_var = st.selectbox("Variable Y", available_y_vars, index=0 if available_y_vars else None)
        
        # Choix d'une variable catégorielle pour la couleur
        cat_cols = [None] + [col for col in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[col]) and col != 'date_hour']
        color_var = st.selectbox("Variable de couleur (optionnel)", cat_cols)
        
        # Graphique de dispersion
        if y_var:
            fig = Visualizations.scatter_plot(filtered_df, x_var, y_var, color_var)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcul de la corrélation
            if color_var is None:
                corr = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
                
                st.markdown(f"### Corrélation entre {x_var} et {y_var}")
                
                if abs(corr) < 0.2:
                    corr_strength = "très faible"
                elif abs(corr) < 0.4:
                    corr_strength = "faible"
                elif abs(corr) < 0.6:
                    corr_strength = "modérée"
                elif abs(corr) < 0.8:
                    corr_strength = "forte"
                else:
                    corr_strength = "très forte"
                
                st.markdown(f"Coefficient de corrélation: **{corr:.4f}** (corrélation {corr_strength})")
                
                if corr > 0:
                    st.markdown("☝️ Les variables évoluent dans le **même sens**.")
                elif corr < 0:
                    st.markdown("👇 Les variables évoluent en **sens opposé**.")
                else:
                    st.markdown("➡️ Pas de relation linéaire détectée entre les variables.")
    
    elif viz_options == "Comparaison par catégorie":
        # Variables catégorielles disponibles
        cat_cols = [col for col in filtered_df.columns if not pd.api.types.is_numeric_dtype(filtered_df[col]) and col != 'date_hour']
        if not cat_cols:
            st.warning("Aucune variable catégorielle disponible dans les données filtrées.")
            return
        
        cat_var = st.selectbox("Variable catégorielle", cat_cols)
        
        # Variables numériques disponibles
        num_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
        metric = st.selectbox("Métrique à analyser", num_cols)
        
        # Limiter le nombre de catégories affichées
        cat_counts = filtered_df[cat_var].value_counts()
        if len(cat_counts) > 15:
            st.warning(f"Il y a {len(cat_counts)} catégories différentes. Seules les 15 plus fréquentes sont affichées.")
            top_cats = cat_counts.nlargest(15).index
            display_df = filtered_df[filtered_df[cat_var].isin(top_cats)]
        else:
            display_df = filtered_df
        
        # Graphique en barres
        fig = Visualizations.aggregate_bar_chart(display_df, cat_var, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau de statistiques par catégorie
        st.markdown("### Statistiques par catégorie")
        stats_by_cat = display_df.groupby(cat_var)[metric].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        stats_by_cat.columns = [cat_var, 'Moyenne', 'Écart-type', 'Minimum', 'Maximum', 'Nombre']
        st.dataframe(stats_by_cat.sort_values('Moyenne', ascending=False), hide_index=True)
    
    elif viz_options == "Carte de chaleur":
        # Variables pour les axes X et Y
        cols_for_heatmap = [col for col in filtered_df.columns if col != 'date_hour']
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Axe X", cols_for_heatmap, index=0)
        
        with col2:
            remaining_cols = [col for col in cols_for_heatmap if col != x_col]
            y_col = st.selectbox("Axe Y", remaining_cols, index=0 if remaining_cols else None)
        
        # Variables pour la valeur
        num_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
        value_col = st.selectbox("Valeur à représenter", num_cols)
        
        # Limiter le nombre de catégories
        if not pd.api.types.is_numeric_dtype(filtered_df[x_col]):
            x_counts = filtered_df[x_col].value_counts()
            if len(x_counts) > 20:
                st.warning(f"L'axe X contient {len(x_counts)} catégories. Seules les 20 plus fréquentes sont utilisées.")
                top_x = x_counts.nlargest(20).index
                filtered_df = filtered_df[filtered_df[x_col].isin(top_x)]
        
        if not pd.api.types.is_numeric_dtype(filtered_df[y_col]):
            y_counts = filtered_df[y_col].value_counts()
            if len(y_counts) > 20:
                st.warning(f"L'axe Y contient {len(y_counts)} catégories. Seules les 20 plus fréquentes sont utilisées.")
                top_y = y_counts.nlargest(20).index
                filtered_df = filtered_df[filtered_df[y_col].isin(top_y)]
        
        # Créer et afficher la carte de chaleur
        if not filtered_df.empty:
            try:
                fig = Visualizations.heatmap(filtered_df, x_col, y_col, value_col)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Impossible de générer la carte de chaleur: {str(e)}")
                st.markdown("""
                Conseils:
                - Vérifiez que vos axes ont suffisamment de valeurs différentes
                - Essayez d'autres combinaisons de variables
                - Les variables avec trop de valeurs uniques peuvent causer des problèmes
                """)
        else:
            st.warning("Pas assez de données pour créer une carte de chaleur avec les critères sélectionnés.")
    
    # Tableau de données filtrées (avec option d'expansion)
    with st.expander("Voir les données filtrées", expanded=False):
        st.dataframe(filtered_df, height=300)
        
        # Option pour télécharger les données filtrées
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"donnees_filtrees_{st.session_state.aggregation_level}.csv", mime='text/csv')
              
