FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

COPY ./data/ /input
CMD cp -a /input/. /data/output

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install -r /input/requirements.txt


