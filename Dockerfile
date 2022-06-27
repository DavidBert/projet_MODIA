
FROM Python:3.8
WORKDIR /usr/src/ app
COPY recommender_app.py  .
COPY model_randomforest.pkl .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
CMD ["python","./recommender_app.py"]
