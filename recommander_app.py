import gradio as gr
from tensorflow.keras.models import load_model
from nltk import word_tokenize          

import pickle
import numpy as np

class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`','!', '#', '$', '%', '&', "'", "'m", '(', ')', '*', '-', '--', '..',
       '...', '....', '.....', '......', '...........', '.i', '.it',
       '.zaar', '/', '039', '08', '1', '1-2', '1.5', '1/2', '1/2tsp',
       '1/3', '1/4', '1/8/09', '10', '10.', '10/18/10', '103024', '10oz',
       '1108690', '114718', '12', '12/22/08', '13', '14', '146530', '15',
       '154943.', '17th', '18', '1970', '1st', '1t', '2', '2-4', '2.',
       '2/3', '20', '20-30', '200', '2007.', '2008', '2009', '2010',
       '2011', '2012', '2013', '2017', '2018', '211485', '221973',
       '2bleu', '2tbs', '3', '3-4', '3/4', '30', '32', '326538', '35',
       '350f', '392938', '4', '4-5', '4.5', '40', '425', '425f', '462620',
       '5', '5-6', '55856', '5th', '6', '680grms', '7', '7.5', '75',
       '79-year-old', '8', '81194', '9','9-inch', '9.', '95', '9x13', '9x13x2', '<', '=', '=-', '>', '?',
       '@', '[', ']', 'zwt0',
       'zwt5', 'zwt6', 'zwt8', '~', '’', '“', '”']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

def make_pred_from_text(text):
    model = load_model('model_prediction_review.h5')
    with open('tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    if type(text) == str:
        text = np.array([text])
    tfidf_data = tfidf.transform(text).toarray()
    pred = model.predict(tfidf_data)
    if pred[0][1] > 0.5:
        return 'Positive, confidence (%.2f)' % pred[0][1]
    else:
        return 'Negative, confidence (%.2f)' % pred[0][0]

gr.Interface(fn=make_pred_from_text, inputs=["text"], outputs=["textbox"]).launch()