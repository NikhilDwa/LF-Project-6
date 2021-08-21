import numpy as np
import pandas as pd
from src.clean_text import clean_dataframe
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def load_csv():

    """
        Load the csv file.

        Returns
        -------
        [dataframe]
            [returns a dataframe with cleaned words.]
    """

    df = pd.read_csv('./data/questions.csv')
    new_df = clean_dataframe(df)

    return new_df


def finding_cosine_scores(df, text):

    """
        Takes dataframe and the input text to find cosine scores.

        Returns
        -------
        [Dictionary]
            [returns Dictionary of cosine scores; key as index and value as cosine scores.]
    """

    scores = {}
    for index in range(len(df['cleaned_title'])):
        df_embedding = model.encode(df['cleaned_title'][index], convert_to_numpy=True)
        text_embedding = model.encode(text, convert_to_numpy=True)
        cosine_scores = util.cos_sim(df_embedding, text_embedding)
        scores[index] = cosine_scores

    return scores


def top_five_questions_index(all_similarity_scores):

    """
        Takes all the similarity scores. Sort and reverse them.

        Returns
        -------
        [List]
            [returns list of index of top five similar questions.]
    """

    similarity_scores_index = sorted(all_similarity_scores, key=all_similarity_scores.get)
    reverse_similarity_scores_index = similarity_scores_index[::-1]
    top_five_similarity_scores_index = reverse_similarity_scores_index[:5]

    return top_five_similarity_scores_index
