import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class QuestionnaireAnalysis:
    def __init__(self, data_fname):
        """
        Initialize with a filename or Path to the JSON data file.
        """
        # Accept string or pathlib.Path
        if not isinstance(data_fname, (str, pathlib.Path)):
            raise TypeError("data_fname must be a path or string")
        path = pathlib.Path(data_fname)
        if not path.is_file():
            raise ValueError(f"File not found: {data_fname}")
        self.data_fname = path
        self.data = None

    def read_data(self):
        """
        Reads the JSON data into a pandas DataFrame and stores it in self.data.
        Converts age and question columns to numeric, coercing errors to NaN.
        """
        with self.data_fname.open() as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Convert relevant columns to numeric types
        cols = ['age', 'q1', 'q2', 'q3', 'q4', 'q5']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        self.data = df

    def show_age_distrib(self) -> tuple:
        """
        Calculates and plots the age distribution with bins [0,10), [10,20), ..., [90,100].
        Returns:
            hist (np.ndarray): counts per bin
            bins (np.ndarray): bin edges
        """
        ages = self.data['age']
        bins = np.arange(0, 101, 10)
        hist, edges = np.histogram(ages, bins=bins)
        # Plot for visualization
        plt.figure()
        plt.hist(ages.dropna(), bins=bins)
        return hist, edges

    def remove_rows_without_mail(self) -> pd.DataFrame:
        """
        Filters out rows whose 'email' field does not meet basic validity checks:
        - Exactly one '@', not at the start or end
        - Contains at least one '.', not at the start or end
        - The domain (after '@') does not start or end with a dot
        Returns a reset-index DataFrame of valid entries.
        """
        df = self.data.copy()
        def valid(email):
            if not isinstance(email, str):
                return False
            if email.count('@') != 1:
                return False
            if email.startswith('@') or email.endswith('@'):
                return False
            if '.' not in email:
                return False
            if email.startswith('.') or email.endswith('.'):
                return False
            local, domain = email.split('@')
            if domain.startswith('.') or domain.endswith('.'):
                return False
            return True
        mask = df['email'].apply(valid)
        return df[mask].reset_index(drop=True)

    def fill_na_with_mean(self) -> tuple:
        """
        Finds rows with any NaN in questions q1–q5, replaces each NaN with
        the mean of the other answered questions for that subject.
        Returns:
            df (pd.DataFrame): corrected DataFrame with filled values
            rows (np.ndarray): indices of rows that had at least one NaN
        """
        df = self.data.copy()
        q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']
        # Identify rows with any missing answers
        na_mask = df[q_cols].isna().any(axis=1)
        rows = np.where(na_mask)[0]
        for i in rows:
            mean_val = df.loc[i, q_cols].mean(skipna=True)
            df.loc[i, q_cols] = df.loc[i, q_cols].fillna(mean_val)
        return df, rows

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        """
        Computes an integer 'score' for each subject as the floor of the mean
        of q1–q5, provided they have at most 'maximal_nans_per_sub' missing;
        otherwise assigns NA.
        Returns a new DataFrame with a 'score' column of dtype UInt8.
        """
        df = self.data.copy()
        q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']
        # Count missing answers per subject
        missing_counts = df[q_cols].isna().sum(axis=1)
        raw_means = df[q_cols].mean(axis=1)
        # Subjects with too many NaNs get NaN score
        raw_means[missing_counts > maximal_nans_per_sub] = np.nan
        # Floor the means, preserving NaN
        scores = []
        for val in raw_means:
            if pd.isna(val):
                scores.append(pd.NA)
            else:
                scores.append(math.floor(val))
        # Create UInt8 series
        df['score'] = pd.Series(scores, index=df.index, dtype='UInt8')
        return df

    def correlate_gender_age(self) -> pd.DataFrame:
        """
        Groups subjects by gender and a boolean indicating age > 40,
        then returns the average q1–q5 scores per group as a DataFrame
        with a MultiIndex (gender, age).
        """
        df = self.data.copy()
        # Drop subjects without a valid age
        df = df.dropna(subset=['age'])
        q_cols = ['q1', 'q2', 'q3', 'q4', 'q5']
        # Boolean age grouping: True if age > 40
        age_bool = df['age'] > 40
        age_bool.name = 'age'
        # Compute mean scores per group
        grouped = df.groupby(['gender', age_bool])[q_cols].mean()
        return grouped
