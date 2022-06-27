import gradio as gr
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import pickle


# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [",", ".", ";", ":", '"', "``", "''", "`"]

    def __init__(self):
        self.stemmer = SnowballStemmer("english")

    def __call__(self, doc):
        return [
            self.stemmer.stem(t)
            for t in word_tokenize(doc)
            if t not in self.ignore_tokens
        ]


tokenizer = StemTokenizer()


def is_it_good(comment: str):
    model_name = "model_randomforest.pkl"
    model = pickle.load(open(model_name, "rb"))

    if type(comment) is str:
        comment = [comment]
    sentiment = model.predict(comment)
    if sentiment == 1:
        to_return = f"positive comment!"
    else:
        to_return = f"negative comment! "
    return to_return


gr.Interface(
    fn=is_it_good,
    inputs=["text"],
    outputs=["textbox"],
    server_port=7860,
    server_name="0.0.0.0",
).launch(
    server_port=7860, server_name="0.0.0.0"
)  # share=true
