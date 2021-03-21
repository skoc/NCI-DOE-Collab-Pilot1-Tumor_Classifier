FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y curl apt-utils
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install -y vim make gcc wget tar unzip git

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt-get update && apt-get install -y python3.8 python3-pip
# RUN apt-get update && apt-get install -y python3-venv python3-dev

# RUN python3 -m pip install --upgrade pip && python3 -m pip install -U tensorflow==2.4
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# install miniconda
ENV CONDA_DIR=/root/miniconda3
ENV PATH=${CONDA_DIR}/bin:${PATH}
ARG PATH=${CONDA_DIR}/bin:${PATH}
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

#RUN apt-get update && apt-get install -y nvidia-smi
RUN python3 -m pip install --upgrade pip && python3 -m pip install pandas numpy sklearn
RUN python3 -m pip install --upgrade pip && python3 -m pip install matplotlib seaborn
RUN python3 -m pip install --upgrade pip && python3 -m pip install -U slideio

WORKDIR /opt

COPY Dockerfile /opt
COPY environment.yml /opt
COPY common /opt/common

# install conda env
RUN conda config --set ssl_verify no
RUN conda env create -f environment.yml -n TC1 \
    && conda init bash \
    && echo "conda activate TC1" >> /root/.bashrc

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "TC1", "/bin/bash", "-c"]

# The code to run when container is started:
# ENTRYPOINT ["conda", "run", "-n", "TC1"]
