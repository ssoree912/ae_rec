FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN conda install -y -n base -c conda-forge mamba \
    && conda clean -afy

WORKDIR /workspace/AlignedForensics

COPY environment.yml /tmp/environment.yml
RUN sed '/^prefix:/d' /tmp/environment.yml > /tmp/environment.docker.yml \
    && sed -i -E '/^[[:space:]]*-[[:space:]]*(openssl|cryptography)=/d' /tmp/environment.docker.yml \
    && sed -i -E 's/^([[:space:]]*-[[:space:]]*[^=[:space:]]+=[^=[:space:]]+)=.*/\1/' /tmp/environment.docker.yml

RUN mamba env create -n DMIDetection -f /tmp/environment.docker.yml \
    && conda clean -afy

COPY requirements.txt /tmp/requirements.txt
RUN conda run -n DMIDetection pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace/AlignedForensics

ENV PATH=/opt/conda/envs/DMIDetection/bin:/opt/conda/bin:${PATH} \
    CONDA_DEFAULT_ENV=DMIDetection

CMD ["/bin/bash"]
