
FROM python:3.8
RUN cd /home/ && mkdir projet_ia 
WORKDIR /home/projet_ia
COPY / .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt
ENV PATH=/root/.local:$PATH
EXPOSE 7860
CMD ["python","./recommender_app.py"]
