"""
AgroQA Farmer Chatbot Engine
Shared logic for Google Colab training and Streamlit deployment.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_THRESHOLD = 0.12


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


class AgroChatbot:
    """Question-answer chatbot backed by TF-IDF similarity over AgroQA dataset."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df = self._load_dataset(self.csv_path)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_text"])

    @staticmethod
    def _load_dataset(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = {"Crop", "Question", "Answer"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        df = df.dropna(subset=["Question", "Answer"]).copy()
        df["Crop"] = df["Crop"].astype(str).str.strip().str.lower()
        df["Question"] = df["Question"].astype(str).str.strip()
        df["Answer"] = df["Answer"].astype(str).str.strip()
        df["search_text"] = (
            df["Crop"] + " " + df["Question"].map(clean_text)
        ).str.strip()
        return df.reset_index(drop=True)

    def get_answer(
        self,
        question: str,
        crop_filter: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Tuple[str, Optional[str], float, Optional[str]]:
        """
        Returns: (answer, matched_crop, confidence_score, matched_question)
        """
        if not question or not question.strip():
            return (
                "Please ask a farming question by voice or text.",
                None,
                0.0,
                None,
            )

        query = clean_text(question)
        crop = (crop_filter or "all").lower().strip()

        if crop != "all":
            mask = self.df["Crop"] == crop
            if mask.any():
                subset = self.df[mask]
                matrix = self.tfidf_matrix[mask.values]
            else:
                subset = self.df
                matrix = self.tfidf_matrix
            query = f"{crop} {query}"
        else:
            subset = self.df
            matrix = self.tfidf_matrix

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix).flatten()

        best_local_idx = int(scores.argmax())
        best_score = float(scores[best_local_idx])
        row = subset.iloc[best_local_idx]

        if best_score < threshold:
            return (
                "I could not find a close match. Try rephrasing your question, "
                "or select a crop (Maize, Beans, Cassava, General).",
                None,
                best_score,
                None,
            )

        return row["Answer"], row["Crop"], best_score, row["Question"]

    def save(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "df": self.df,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "csv_path": str(self.csv_path),
        }
        with open(model_path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, model_path: str | Path) -> "AgroChatbot":
        model_path = Path(model_path)
        with open(model_path, "rb") as handle:
            payload = pickle.load(handle)

        bot = cls.__new__(cls)
        bot.csv_path = Path(payload.get("csv_path", "AgroQA Dataset.csv"))
        bot.df = payload["df"]
        bot.vectorizer = payload["vectorizer"]
        bot.tfidf_matrix = payload["tfidf_matrix"]
        return bot

    @property
    def crop_options(self) -> list[str]:
        crops = sorted(self.df["Crop"].unique().tolist())
        return ["all"] + crops

    @property
    def stats(self) -> dict:
        return {
            "total_qa_pairs": len(self.df),
            "crops": self.df["Crop"].value_counts().to_dict(),
        }
