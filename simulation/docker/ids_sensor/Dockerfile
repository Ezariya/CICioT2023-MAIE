# simulation/docker/ids_sensor/Dockerfile
# IDS sensor: Python + scapy + scikit-learn + joblib.
# Mounts the repo at /workspace so it can use the trained models.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tcpdump tshark iproute2 iputils-ping net-tools libpcap0.8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        scapy==2.5.0 \
        pyspark==3.5.0 \
        pandas==2.1.* \
        numpy==1.26.* \
        scikit-learn==1.3.* \
        xgboost==2.0.* \
        lightgbm==4.1.* \
        joblib==1.3.* \
        pyyaml==6.* \
        streamlit==1.29.* \
        plotly==5.18.* \
        psutil==5.9.*

WORKDIR /workspace
ENV PYTHONPATH=/workspace
CMD ["sleep", "infinity"]
