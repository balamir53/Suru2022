FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /vasr/lib/apt/lists/*
RUN apt-get update

RUN mkdir /app
# Create a non-root user and switch to it
# chown?
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.9.5 \
 && conda clean -ya

RUN conda install pytorch==1.12 cudatoolkit=10.2 -c pytorch -y
RUN pip install tensorflow==2.8.0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# COPY data/inputs /data/inputs
COPY data /data
WORKDIR /data

ENTRYPOINT ["python3", "test.py"]