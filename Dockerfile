From pytorch/pytorch

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install ipykernel scikit-learn lime pandas numpy matplotlib plotly seaborn nltk nbformat gradio jupyter
RUN python -m nltk.downloader punkt
