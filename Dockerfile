FROM ubuntu:18.04
MAINTAINER tomas.prinda@gmail.com

USER root
ENV SHELL /bin/bash

# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    git \
    vim \
    wget \
    build-essential \
    python-dev \
    ca-certificates \
    bzip2 \
    unzip \
    libsm6 \
    sudo \
    locales \
    libzmq3-dev \
    python3-pip \
    python3-dev \
    && apt-get clean
    
RUN apt-get install -yq --no-install-recommends \
    graphviz \
    python3-setuptools

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'

# install pip packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade jupyter
RUN pip3 install --upgrade matplotlib
RUN pip3 install --upgrade numpy
RUN pip3 install --upgrade click
RUN pip3 install --upgrade pandas
RUN pip3 install --upgrade scikit-learn
RUN pip3 install --upgrade pytest
RUN pip3 install --upgrade graphviz
RUN pip3 install --upgrade xgboost
RUN pip3 install --upgrade scikit-surprise
RUN pip3 install --upgrade pdpbox
RUN pip3 install --upgrade shap
RUN pip3 install --upgrade torch
RUN pip3 install --upgrade torchvision 
RUN pip3 install --upgrade bokeh 


# Data
# originally from https://www.kaggle.com/CooperUnion/cardataset
RUN mkdir /data
RUN wget -O /data/cardataset.zip http://lezo.cz/no_drupal/cardataset.zip

# Flexp
EXPOSE 7777
RUN git clone https://github.com/seznam/flexp.git
WORKDIR /flexp
RUN python3 setup.py develop  

# tputils
WORKDIR /
RUN git clone https://github.com/tomasprinda/tputils.git
WORKDIR /tputils
RUN python3 setup.py install  


# Jupyter notebook
EXPOSE 8888
RUN mkdir /root/.jupyter/
COPY jupyter_notebook_config.py /root/.jupyter/  
# Solves Kernel crashes with ModuleNotFoundError on 'prompt_toolkit.formatted_text'
#by https://github.com/jupyter/notebook/issues/4050
RUN pip install git+https://github.com/jupyter/jupyter_console

# will be mounted to master machine
RUN mkdir /src
WORKDIR /src
