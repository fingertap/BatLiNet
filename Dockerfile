FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /root

COPY . /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends ttf-mscorefonts-installer && \
    fc-cache -f && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install jupyter ipykernel
