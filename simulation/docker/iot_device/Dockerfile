# simulation/docker/iot_device/Dockerfile
# Lightweight IoT-device simulator container.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tcpdump iproute2 iputils-ping net-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir scapy==2.5.0

WORKDIR /app
COPY sim_device.py /app/sim_device.py

CMD ["python", "/app/sim_device.py"]
