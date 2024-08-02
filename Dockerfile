FROM nvidia/cuda:12.2.0-base-ubuntu22.04
COPY . /app
RUN apt update
RUN apt upgrade -y
RUN apt install -y vim python3 python3-pip
RUN pip install pyRealParser torch tokenizers pandas numpy rotary_embedding_torch
