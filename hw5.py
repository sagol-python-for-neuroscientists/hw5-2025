import json
import pathlib
from typing import Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class QuestionnaireAnalysis:

    def __init__(self, data_fname: Union[pathlib.Path, str]):
        if not isinstance(data_fname, (str, pathlib.Path)):
            raise TypeError("data_fname must be a string or pathlib.Path")
        self.data_fname = pathlib.Path(data_fname)
        if not self.data_fname.exists():
            raise ValueError(f"File does not exist: {self.data_fname}")
        self.data = None

    def read_data(self):
        with open(self.data_fname, "r", encoding="utf-8") as f:
            self.data = pd.DataFrame(json.load(f))

    # --------------------------------------------
    # Question 1: Plot Age Distribution Histogram
    # --------------------------------------------

    def show_age_distrib(self) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.data is None:
            raise ValueError("Data not loaded. Call `read_data()` first.")

        ages = pd.to_numeric(self.data["age"], errors="coerce")
        valid_ages = ages[(ages >= 0) & (ages <= 120)]

        bins = np.arange(0, 101, 10)
        hist, bin_edges = np.histogram(valid_ages, bins=bins)

        plt.figure(figsize=(8, 5))
        plt.hist(valid_ages, bins=bins, edgecolor='black', rwidth=0.9)
        plt.title("Age Distribution of Participants")
        plt.xlabel("Age")
        plt.ylabel("Number of Participants")
        plt.grid(True)
        plt.show()

        return hist, bin_edges
    # -----------------------------------------------
    # Question 2: Remove Rows with Invalid Email Addr
    # -----------------------------------------------

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
            if '.' not in email:
                return False
            local, domain = email.split('@')
            if not local or not domain:
                return False
            if domain.startswith('.'):
                return False
            return True

        mask = self.data['email'].apply(is_valid_email)
        cleaned_df = self.data[mask].reset_index(drop=True)
        return cleaned_df

    # ------------------------------------------------------------------------------
    # Question 3: Fill Missing Answers with Subject's Average of Other Responses
    # ------------------------------------------------------------------------------

    def fill_na_with_mean(self) -> Tuple[pd.DataFrame, np.ndarray]:
        question_cols = [col for col in self.data.columns if col.startswith("q")]
        df = self.data.copy()
        modified_indices = []

        for idx, row in df.iterrows():
            row_values = row[question_cols]
            row_values = pd.to_numeric(row_values, errors="coerce")  # 👈 Ensure numeric
            na_mask = row_values.isna()

            if na_mask.any():
                mean_val = row_values[~na_mask].mean()
                df.loc[idx, na_mask.index[na_mask]] = mean_val
                modified_indices.append(idx)

        return df, np.array(sorted(modified_indices), dtype=int)

# ------------------------------------------------------------------
# Question 4: Calculate Scores with Max Allowed Missing Responses
# ------------------------------------------------------------------

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        grade_cols = ["q1", "q2", "q3", "q4", "q5"]

        grade_vals = self.data[grade_cols].apply(pd.to_numeric, errors="coerce")
        nan_counts = grade_vals.isna().sum(axis=1)

        # Compute score
        score = np.floor(grade_vals.mean(axis=1))

        # Set NaN where too many values are missing
        score[nan_counts > maximal_nans_per_sub] = pd.NA

        # Cast to nullable UInt8
        score = score.astype("UInt8")

        df = self.data.copy()
        df["score"] = score

        return df


    
    def correlate_gender_age(self) -> pd.DataFrame:
        df = self.data.copy()

        question_cols = [col for col in df.columns if col.startswith('q')]
        df[question_cols] = df[question_cols].apply(pd.to_numeric, errors='coerce')
        
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

        df = df[df['gender'].notna() & df['age'].notna()]

        df['age'] = df['age'] > 40  
        
        df[question_cols] = df.groupby(['gender', 'age'])[question_cols].transform(lambda x: x.fillna(x.mean()))

        grouped = df.groupby(['gender', 'age'])[question_cols].mean()

        return grouped.sort_index()