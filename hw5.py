from __future__ import annotations

import json
import pathlib
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QuestionnaireAnalysis:
    def __init__(self, data_fname: Union[pathlib.Path, str]):
        self.data_fname = pathlib.Path(data_fname)
        self.data: pd.DataFrame | None = None

    def read_data(self) -> None:
        self.data = pd.read_json(self.data_fname)

# 1. Plotting the distribution of ages of the participants
    
    def show_age_distrib(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise RuntimeError("run read_data() first")
        ages = self.data["age"].to_numpy()
        bins = np.arange(0, 101, 10)
        hist, bins = np.histogram(ages, bins=bins)

        plt.figure(figsize=(8, 4))
        plt.bar(bins[:-1], hist, width=8, align="edge")
        plt.xlabel("Age")
        plt.ylabel("Number of participants")
        plt.title("Age distribution")
        plt.xticks(bins)
        plt.tight_layout()
        plt.show()

        return hist, bins

# 2. Removing the rows with an invalid address
    
    @staticmethod
    def _is_valid_email(addr: str) -> bool:
        if not isinstance(addr, str):
            return False

        if addr.startswith("@") or addr.endswith("@"):
            return False
        if addr.startswith(".") or addr.endswith("."):
            return False
        if addr.count("@") != 1:
            return False
        local, domain = addr.split("@")
        if domain.startswith("."):
            return False
        if "." not in domain:
            return False
        return True

    def remove_rows_without_mail(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("run read_data() first")

        mask = self.data["email"].apply(self._is_valid_email)
        cleaned = self.data.loc[mask].reset_index(drop=True)
        return cleaned
        
# 3. Replacing the missing values with the mean for the subject 
    
    def _question_columns(self) -> List[str]:
        if self.data is None:
            raise RuntimeError("run read_data() first")
        numeric_cols = self.data.select_dtypes(include="number").columns
        grade_cols = [c for c in numeric_cols if c.lower().startswith("q")]
        return grade_cols

    def fill_na_with_mean(self) -> Tuple[pd.DataFrame, np.ndarray]:
        if self.data is None:
            raise RuntimeError("run read_data() first")
        df = self.data.copy()
        grade_cols = self._question_columns()
        modified_rows: List[int] = []
        for idx, row in df[grade_cols].iterrows():
            if row.isna().any():
                mean_grade = row.mean(skipna=True)
                df.loc[idx, grade_cols] = row.fillna(mean_grade)
                modified_rows.append(idx)

        return df, np.array(modified_rows, dtype=int)
        
# 4.  Adding the score column

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("run read_data() first")
        df = self.data.copy()
        grade_cols = self._question_columns()
        nan_count = df[grade_cols].isna().sum(axis=1)
        avg_scores = np.floor(df[grade_cols].mean(axis=1)).astype("UInt8")
        avg_scores = avg_scores.mask(nan_count > maximal_nans_per_sub, pd.NA)

        df["score"] = avg_scores.astype("UInt8")

        return df

# 5.  Exploring the correlation between the subject's gender, age and grades

    def correlate_gender_age(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("run read_data() first")
        df = self.data.copy()
        grade_cols = self._question_columns()
        df = df.set_index(["gender", "age"], append=True)
        df["age_above_40"] = df.index.get_level_values("age") > 40
        grouped = (
            df.groupby(["gender", "age_above_40"])[grade_cols]
            .mean()
            .astype(float)
        )

        return grouped