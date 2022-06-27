
FROM python:3.8
WORKDIR /code
COPY recommender_app.py  .
COPY requirement.txt .
COPY model_randomforest.pkl .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gradio nltk
ENV PATH=/root/.local:$PATH
EXPOSE 7860
CMD ["python","./recommender_app.py"]
