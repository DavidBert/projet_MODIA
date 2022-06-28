import gradio as gr
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import pickle
import unicodedata
import re
from gensim.models import Word2Vec
import numpy as np

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def set_features(array_token_prod, dim_feature, model):
    features = []
    # on prend les tokens de chaque produit
    for token_prod in array_token_prod:
        # on crée le np final du produit
        feat_prod = np.zeros(dim_feature)
        # on prend chaque token dans la desc prod
        for token in token_prod:
            # on somme les vecteurs et on divise direct par le nombre de token
            feat_prod += model[token] / len(token_prod)

        # on rajoute le vectuer moyenné produit dans la list features
        features += [feat_prod]
    return features


def cleaning(voca):

    # STEP 1 : removing special characters and put the vocabulary in lower
    cleaned_voc = voca
    cleaned_voc = cleaned_voc.lower()
    cleaned_voc = cleaned_voc.replace("\u2026", ".")
    cleaned_voc = cleaned_voc.replace("\u00a0", " ")

    # STEP 2 : takes off some punctuation
    cleaned_voc = (
        unicodedata.normalize("NFD", cleaned_voc)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    # STEP 3 : keeps only alphabet letters
    cleaned_voc = re.sub("[^a-z_]", " ", cleaned_voc)

    # removing what in the stopwords ??? punctuation ?
    stopwords = [
        unicodedata.normalize("NFD", sw).encode("ascii", "ignore").decode("utf-8")
        for sw in stop_words
    ]

    tokens = [
        w for w in cleaned_voc.split(" ") if (len(w) > 2) and (w not in stopwords)
    ]
    removed_words = [
        w for w in cleaned_voc.split(" ") if (len(w) < 2) or (w in stopwords)
    ]

    ## Stemming function to get roots of words (racines des mots)
    stemmer = nltk.stem.SnowballStemmer("english")
    tokens_stem = [stemmer.stem(token) for token in tokens]

    return tokens_stem


def is_it_good(comment: str):
    model_name = "reg_log.pkl"
    # model_name = "model_randomforest.pkl"
    if (
        model_name[-3:] == "pkl"
    ):  # on peut modifier le nom afin de changer de model !!attention cela modifiera aussi le model pour le dockerfile en cas de dockerfile!!!
        model = pickle.load(open(model_name, "rb"))
    model = Word2Vec.load("word2vec.model")

    comment = cleaning(comment)
    comment = [" ".join(comment)]
    if model_name == "reg_log.pkl":
        comment = set_features(comment, model.vector_size, model)

    sentiment = model.predict(comment)
    if sentiment == 1:
        to_return = f"positive comment!"
    else:
        to_return = f"negative comment! "
    return to_return


is_it_good("I love this recipe")
gr.Interface(
    fn=is_it_good,
    inputs=["text"],
    outputs=["textbox"],
    server_port=7860,
    server_name="0.0.0.0",
).launch(server_port=7860, server_name="0.0.0.0")
