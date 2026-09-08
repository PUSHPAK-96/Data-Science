
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def topic_labels(texts, n_topics=5):
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=2)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_topics, n_init=10, random_state=42).fit(X)
    return km.labels_
