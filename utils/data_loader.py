import pandas as pd
from pathlib import Path

class DNSDataLoader:
    def __init__(self):
        self.data_path = Path(
            r"sample_data.parquet"
        )
        
    def load_data(self):
        data = pd.read_parquet(self.data_path)
        data['date_hour'] = pd.to_datetime(data['date_hour'])
        return data
    
    def load_aggregated_data(self, aggregation_level, eda_type=None):
        """Charge et agrège les données selon le niveau spécifié et le type d'analyse"""
        df = self.load_data()
        
        # Définir les colonnes d'agrégation
        if aggregation_level == "peag_nro":
            group_cols = ["peag_nro", "date_hour"]
        elif aggregation_level == "olt_name":
            group_cols = ["olt_name", "date_hour"]
        elif aggregation_level == "peag_nro & olt_name":
            group_cols = ["peag_nro", "olt_name", "date_hour"]
        else:
            return df

        # Colonnes additionnelles
        additional_group_cols = []
        for col in ["code_departement", "olt_model", "boucle", "dsp", "pebib"]:
            if col in df.columns and col not in group_cols:
                additional_group_cols.append(col)

        group_cols.extend(additional_group_cols)

        # Agrégation
        df_agg = df.groupby(group_cols, as_index=False).agg({
            "avg_dns_time": "mean",
            "std_dns_time": "mean",
            "nb_test_dns": "sum",
            "nb_client_total": "sum",
            "pop_dns": "first" if "pop_dns" in df.columns else None,
            "nb_test_scoring": "sum" if "nb_test_scoring" in df.columns else None,
            "avg_latence_scoring": "mean" if "avg_latence_scoring" in df.columns else None,
            "std_latence_scoring": "mean" if "std_latence_scoring" in df.columns else None,
            "avg_score_scoring": "mean" if "avg_score_scoring" in df.columns else None,
            "std_score_scoring": "mean" if "std_score_scoring" in df.columns else None
        })

        df_agg = df_agg.loc[:, ~df_agg.columns.isnull()]

        # 🎯 Filtrage selon le type d'EDA
        if eda_type == "EDA DNS" and "nb_test_dns" in df_agg.columns:
            df_agg = df_agg[df_agg["nb_test_dns"] > 0]
        elif eda_type == "EDA SCORING" and "nb_test_scoring" in df_agg.columns:
            df_agg = df_agg[df_agg["nb_test_scoring"] > 0]

        return df_agg
