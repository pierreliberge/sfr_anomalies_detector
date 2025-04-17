import pandas as pd
from pathlib import Path

class DNSDataSampler:
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.data = None

    def load_data(self):
        print(f"Chargement des données depuis : {self.input_path}")
        self.data = pd.read_parquet(self.input_path)
        self.data['date_hour'] = pd.to_datetime(self.data['date_hour'])
        print("Date min :", self.data['date_hour'].min())
        print("Date max :", self.data['date_hour'].max())

    def filter_by_date(self, start_date, end_date):
        if self.data is None:
            raise ValueError("Les données doivent être chargées avant de filtrer.")
        filtered = self.data[
            (self.data['date_hour'] >= pd.to_datetime(start_date)) &
            (self.data['date_hour'] <= pd.to_datetime(end_date))
        ]
        print(f"{len(filtered)} lignes sélectionnées entre {start_date} et {end_date}.")
        return filtered

    def export_data(self, df, output_path):
        output_path = Path(output_path)
        df.to_parquet(output_path, index=False)
        print(f"Échantillon sauvegardé dans {output_path}")

if __name__ == "__main__":
    input_file = r"250327_tests_fixe_dns_sah_202412_202501.parquet"
    output_file = r"sample_data.parquet"

    sampler = DNSDataSampler(input_file)
    sampler.load_data()
    sample_df = sampler.filter_by_date("2024-12-01 00:00:00", "2025-01-01 23:00:00")
    sampler.export_data(sample_df, output_file)


