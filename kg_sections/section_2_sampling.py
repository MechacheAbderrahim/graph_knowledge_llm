import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def representative_titles(category_df, sample_size, seed=42):
    titles = category_df["title"].tolist()
    sample_size = min(sample_size, category_df["title"].nunique())

    if sample_size < 2:
        return titles[: max(sample_size, 1)]

    vectors = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(titles)
    sample_size = min(sample_size, vectors.shape[0])
    kmeans = KMeans(n_clusters=sample_size, random_state=seed, n_init=10).fit(vectors)

    selected_titles = []
    for cluster_id in range(sample_size):
        row_indexes = np.where(kmeans.labels_ == cluster_id)[0]
        similarities = cosine_similarity(
            vectors[row_indexes], kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        ).ravel()
        selected_titles.append(titles[row_indexes[similarities.argmax()]])

    return selected_titles
