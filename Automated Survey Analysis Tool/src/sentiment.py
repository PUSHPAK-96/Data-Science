
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

_an = SentimentIntensityAnalyzer()

def sentiment_scores(texts: pd.Series) -> pd.Series:
    return texts.map(lambda t: _an.polarity_scores(t)["compound"])

def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"
