import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def representative_titles(category_df, sample_size, seed=42, text_column="title"):
    titles = category_df[text_column].tolist()
    sample_size = min(sample_size, category_df[text_column].nunique())

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


def representative_titles_by_category(
    categories,
    df,
    name_for_category,
    sample_size,
    seed=42,
    text_column="title",
):
    samples = []

    for category_id in categories:
        category_df = df[df["category_id"] == category_id].reset_index(drop=True)
        if len(category_df) == 0:
            continue

        samples.append(
            {
                "category_id": int(category_id),
                "category_name": name_for_category(category_id),
                "titles": representative_titles(
                    category_df,
                    sample_size,
                    seed=seed,
                    text_column=text_column,
                ),
            }
        )

    return samples
