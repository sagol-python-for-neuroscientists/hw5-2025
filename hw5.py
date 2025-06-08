import json
import pathlib
import sys
from typing import Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QuestionnaireAnalysis:
    def __init__(self, data_fname: Union[pathlib.Path, str]):
        try:
            self.data_fname = pathlib.Path(data_fname)
        except TypeError:
            raise TypeError("data_fname must be a string or a Path object")

        if not self.data_fname.exists():
            raise ValueError("File not found")
        self.data = None

    def read_data(self):
        with open(self.data_fname, 'r') as f:
            self.data = pd.DataFrame(json.load(f))
        self.data.replace("nan", np.nan, inplace=True)

    def show_age_distrib(self) -> Tuple[np.ndarray, np.ndarray]:
        age_data = pd.to_numeric(self.data['age'], errors='coerce').dropna()
        hist, bins = np.histogram(age_data, bins=np.arange(0, 110, 10))
        plt.hist(age_data, bins=np.arange(0, 110, 10), edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.title('Age Distribution')
        plt.savefig('output/age_distribution.png')
        if 'pytest' not in sys.modules:
            plt.show()
        return hist, bins

    def remove_rows_without_mail(self) -> pd.DataFrame:
        def is_valid_email(email: str) -> bool:
            if not isinstance(email, str):
                return False
            if email.startswith('@') or email.endswith('@'):
                return False
            if email.startswith('.') or email.endswith('.'):
                return False
            if email.count('@') != 1:
                return False
            at_index = email.index('@')
            if at_index + 1 >= len(email) or email[at_index + 1] == '.':
                return False
            if '.' not in email:
                return False
            return True

        valid_df = self.data[self.data['email'].apply(is_valid_email)].reset_index(drop=True)
        return valid_df

    def fill_na_with_mean(self) -> Tuple[pd.DataFrame, np.ndarray]:
        df = self.data.copy()
        q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']
        rows_to_correct = df[q_cols].isnull().any(axis=1)
        indices = np.where(rows_to_correct)[0]

        for i in indices:
            row_mean = df.loc[i, q_cols].mean()
            df.loc[i, q_cols] = df.loc[i, q_cols].fillna(row_mean)

        return df, indices

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        df = self.data.copy()
        q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']

        nan_counts = df[q_cols].isnull().sum(axis=1)
        scores = df[q_cols].mean(axis=1)
        scores[nan_counts > maximal_nans_per_sub] = np.nan
        df['score'] = scores.apply(lambda x: np.floor(x) if pd.notna(x) else pd.NA)
        df['score'] = df['score'].astype('UInt8')

        return df

    def correlate_gender_age(self) -> pd.DataFrame:
        df = self.data.copy()
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df.dropna(subset=['age'], inplace=True)
        df['age'] = df['age'].astype(int)
        
        df['age_group'] = df['age'] > 40
        
        q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']
        for col in q_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        grouped = df.groupby(['gender', 'age_group'])[q_cols].mean()
        grouped.index.names = ['gender', 'age']
        return grouped

if __name__ == '__main__':
    analysis = QuestionnaireAnalysis('data.json')
    analysis.read_data()
    analysis.show_age_distrib()
