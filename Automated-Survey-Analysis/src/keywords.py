
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def top_keywords(texts: pd.Series, k=15):
    # 1-2 gram TF-IDF; min_df=2 to reduce noise on small corpora
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(texts)
    scores = X.sum(axis=0).A1
    vocab_inv = {v:k for k,v in vec.vocabulary_.items()}
    idx = scores.argsort()[::-1][:k]
    return [vocab_inv[i] for i in idx]
