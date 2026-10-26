FROM tensorflow/tensorflow:2.13.0-gpu

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    bzip2 \
 && rm -rf /var/lib/apt/lists/*

# 2. ARGs for dynamic host UID and GID
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG USERNAME=tfuser

RUN groupadd -g ${HOST_GID} ${USERNAME} \
 && useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${USERNAME} \
 && mkdir /app \
 && chown -R ${HOST_UID}:${HOST_GID} /app

# 5. Switch to the user
USER ${USERNAME}
# Set home for the next RUN commands, micromamba installs here by default
WORKDIR /home/${USERNAME} 

# 6. Install micromamba and Python *as the user*
SHELL ["/bin/bash", "-c"]
RUN curl -Ls micro.mamba.pm/install.sh | /bin/bash \
  && source ~/.bashrc \
  && /home/${USERNAME}/.local/bin/micromamba install -y python=3.11 \
  && /home/${USERNAME}/.local/bin/micromamba clean --all --yes
SHELL ["/bin/sh", "-c"]

# 7. Go back to the app directory (which you now own)
WORKDIR /app

RUN /home/${USERNAME}/micromamba/bin/python3.11 -m venv env_torch \
      && ./env_torch/bin/pip install --upgrade pip \
      && ./env_torch/bin/pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128
 
RUN ./env_torch/bin/pip install timm

USER root

RUN apt update \
    && apt install -y make \
                       gcc \
                       build-essential \
                       libgtk-3-dev \
                       wget \
                       git \
                       parallel \
                       nano \
                       python3.8-dev \
                       python3.8-venv \
                       python3-pip \
                       openjdk-11-jdk-headless \
                       default-libmysqlclient-dev \
                       libnotify-dev \
                       libsdl2-dev \
                       libwebkit2gtk-4.0-dev

RUN pip install --upgrade pip \
    && pip install wheel cython numpy seaborn \
    && pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04 wxPython==4.2.2 \
    && pip install cellprofiler==4.2.8

# requirements adapted from vqdang/hover_net to make package versions more flexible
COPY requirements.txt .
RUN ./env_torch/bin/pip install -r requirements.txt \
    && git clone https://github.com/vqdang/hover_net \
    && mkdir hovernet_model \
    && pip install gdown \
    && python3 -m gdown https://drive.google.com/uc?id=1NUnO4oQRGL-b0fyzlT8LKZzo6KJD0_6X -O hovernet_model/ \
    && python3 -m gdown https://drive.google.com/uc?id=1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR -O hovernet_model/
# downloads hovernet models kumar and pannuke, respectively

RUN pip install pyopenssl opencv-python docopt --upgrade \
    && apt-get install -y openslide-tools \
    && pip install https://github.com/openslide/openslide-python/releases/download/v1.2.0/openslide-python-1.2.0.tar.gz \
    && pip install havoc-clustering==0.0.20 \
    && wget "https://havoc.s3.amazonaws.com/74_class_model.h5" -P /app/model_cnn/

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gdebi-core \
        libcurl4-openssl-dev \
        libjpeg-dev \
        libmagick++-dev \
        libssl-dev \
        libxml2-dev \
    && export R_VERSION=4.3.2 \
    && curl -O https://cdn.rstudio.com/r/ubuntu-2004/pkgs/r-${R_VERSION}_1_amd64.deb \
    && gdebi -n r-${R_VERSION}_1_amd64.deb \
    && ln -sf /opt/R/${R_VERSION}/bin/R /usr/local/bin/R \
    && ln -sf /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript \
    && rm -f r-${R_VERSION}_1_amd64.deb \
    && rm -rf /var/lib/apt/lists/* \
    && R -e "install.packages(c('docopt', 'reshape2', 'ggplot2', 'readr', 'dplyr', 'data.table', 'remotes', 'cowplot', 'jpeg', 'jsonlite', 'magick', 'paletteer'), repos='https://cloud.r-project.org')" \
    && R -e "remotes::install_version('xgboost', version='1.7.11.1', repos='https://cloud.r-project.org')"

# Some models require auth
# run manually as user tfuser with auth token, and commit image
#USER ${USERNAME}
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    ./env_torch/bin/python -m pip install --no-cache-dir huggingface_hub;\
    ./env_torch/bin/python -c "import os, torch, timm;\
                               from huggingface_hub import login;\
                               login(token=os.environ['HF_TOKEN']);\
                               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');\
                               model = timm.create_model(model_name='hf_hub:prov-gigapath/prov-gigapath', pretrained=True).to(device).eval()"
USER root
# Loads models strictly from local machine, avoids repository checks for newer versions
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY ./src/ /app/src/
COPY ./model_XGBoost/ /app/model_XGBoost/

RUN mkdir -p /app/inputs \
    && mkdir -p /app/outputs \
    && mkdir -p /data/havoc_profiler/svs_files \
    && mkdir -p /data/havoc_profiler/havoc \
    && mkdir -p /data/havoc_profiler/hovernet \
    && mkdir -p /data/havoc_profiler/cellprofiler \
    && mkdir -p /data/havoc_profiler/classifier_region \
    && mkdir -p /data/havoc_profiler/features \
    && mkdir -p /data/havoc_profiler/dlfv \
    && mkdir -p /data/havoc_profiler/dlfv_region \
    && mkdir -p /data/havoc_profiler/risk \
    && mkdir -p /data/havoc_profiler_study \
    && chown -R tfuser:tfuser /data/havoc_profiler \
    && chown -R tfuser:tfuser /data/havoc_profiler_study \
    && chown -R tfuser:tfuser /app/src
    
ENTRYPOINT ["python", "/app/src/app_codido.py"]

# export HF_TOKEN='hf_xxxxx'
# cd /data/docker/havoc-profiler-main
# docker build --secret id=hf_token,env=HF_TOKEN -t havoc-profiler:01 .
