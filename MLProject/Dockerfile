FROM python:3.12.7-slim

WORKDIR /app

COPY MLproject /app/MLproject
COPY modelling.py /app/modelling.py
COPY namadataset_preprocessing /app/namadataset_preprocessing
COPY conda.yaml /app/conda.yaml

RUN python -m pip install --upgrade pip && \
    pip install pandas numpy scikit-learn mlflow==2.19.0 matplotlib seaborn joblib

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

CMD ["python", "modelling.py"]
