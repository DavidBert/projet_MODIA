
FROM python:3.8
WORKDIR /usr/src/app
COPY recommender_app.py  .
COPY requirement.txt .
COPY model_randomforest.pkl .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gradio nltk
CMD ["python","./recommender_app.py"]
