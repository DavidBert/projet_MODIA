import gradio as gr
from tensorflow.keras.models import load_model
import pickle



def make_pred_from_text(text):
    model = load_model('model_prediciton_review.h5')
    with open('tfidf.pkl', 'rb') as file:
        data = pickle.load(f)
    tfidf = pickle.load("tfidf.pkl")
    if type(text) == str:
        text = np.array([text])
    tfidf_data = tfidf.transform(text).toarray()
    return model.predict(tfidf_data)

gr.Interface(fn=make_pred_from_text, inputs=["text"], outputs=["textbox"]).launch()