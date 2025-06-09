import pandas as pd
import numpy as np
from typing import Optional


class QuestionnaireAnalysis:
    def __init__(self, data_fname: str):
        self.data_fname = data_fname
        self.data = None

    def read_data(self):
        """Reads the JSON data located in self.data_fname into memory."""
        self.data = pd.read_json(self.data_fname)

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        """Calculates the average score of a subject and adds a new "score" column
        with it.

        If the subject has more than "maximal_nans_per_sub" NaN in his grades, the
        score should be NA. Otherwise, the score is simply the mean of the other grades.
        The datatype of score is UInt8, and the floating point raw numbers should be
        rounded down.

        Parameters
        ----------
        maximal_nans_per_sub : int, optional
            Number of allowed NaNs per subject before giving a NA score.

        Returns
        -------
        pd.DataFrame
            A new DF with a new column - "score".
        """
        if self.data is None:
            raise ValueError("Data not loaded. Use the read_data method first.")

        # Select columns corresponding to the questions (assuming they are named q1, q2, ..., q5)
        question_columns = [col for col in self.data.columns if col.startswith('q')]

        def calculate_score(row):
            # Count the number of NaNs in the row
            nans_count = row.isna().sum()
            if nans_count > maximal_nans_per_sub:
                return np.nan  # Assign NA if NaNs exceed the threshold
            # Calculate the mean of non-NaN values and round down
            return np.floor(row.dropna().mean())

        # Apply the scoring function to each row
        self.data['score'] = self.data[question_columns].apply(calculate_score, axis=1)

        # Convert the score column to UInt8
        self.data['score'] = self.data['score'].astype(pd.UInt8Dtype())

        return self.data


if __name__ == "__main__":
    analysis = QuestionnaireAnalysis("data.json")
    analysis.read_data()
    scored_data = analysis.score_subjects(maximal_nans_per_sub=1)
    print("Scored Data:")
    print(scored_data)