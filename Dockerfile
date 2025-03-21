FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    sqlite3 \
    libsqlite3-dev \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py39_23.11.0-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py39_23.11.0-1-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN wget -O sqlitestudio.tar.xz https://github.com/pawelsalawa/sqlitestudio/releases/download/3.4.17/sqlitestudio-3.4.17-linux-x64.tar.xz && \
    mkdir -p /opt/sqlitestudio && \
    tar -xf sqlitestudio.tar.xz -C /opt/sqlitestudio --strip-components=1 && \
    rm sqlitestudio.tar.xz

ENV PATH="/opt/sqlitestudio:$PATH"

RUN conda install -y pip && conda clean --all -y

RUN apt-get install -y \
    sqlite3

RUN conda config --add channels conda-forge

RUN pip install --no-cache-dir tensorflow-gpu==2.5.0
RUN pip install --no-cache-dir pandas==1.1.5

RUN conda install -y \
    matplotlib \
    mplfinance \
    pickleshare \
    pillow \
    python-dateutil \
    python-dotenv \
    pytz \
    scikit-learn \
    scipy \
    threadpoolctl \
    tqdm \
    lime \
    "numpy=1.19.5" \
    "pandas=1.1.5" && \
    conda clean --all -y

ENV CONDA_DEFAULT_ENV=base
ENV PATH="/opt/conda/bin:$PATH"

RUN python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
RUN python -c "import numpy as np; print('NumPy:', np.__version__)"
RUN python -c "import pandas as pd; print('Pandas:', pd.__version__)"

WORKDIR /workspace

