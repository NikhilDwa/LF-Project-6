import re
import nltk
from nltk.corpus import stopwords
import nltk


def clean_dataframe(df):

    """
        Cleans the dataframe provided. Removes stop words.

        Returns
        -------
        [dataframe]
            [returns a dataframe with cleaned words.]
    """

    cleaned = []
    for index in range(len(df['Title'])):
        raw_text = df['Title'][index]
        cleaned_words = clean_text(raw_text)
        cleaned.append(cleaned_words[0])
    df['cleaned_title'] = cleaned

    return df


def clean_text(text):

    """
        Cleans the text provided. Removes stop words.

    Returns
    -------
    [list]
        [returns a list of string of cleaned word.]
    """

    text = text.lower()
    text = re.sub(r"\[[0-9]*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.lower().split()
    stop_word_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stop_word_set]))
    cleaned_words = [" ".join(cleaned_words[::])]

    return cleaned_words
