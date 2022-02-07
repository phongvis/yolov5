FROM nvidia/cuda:11.3.0-runtime

# python3, some directly used packages and some just to work around
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y unzip libgl1-mesa-dev libglib2.0-0 wget python3-dev python3 python3-pip python3-setuptools && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install -U pip

WORKDIR /root

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
COPY yolov5 /root
COPY yolov5/rs /root

# Download base model
RUN wget -nv https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l6.pt
RUN wget -nv https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

# # Installs required packages
RUN pip install -r requirements.txt

# Sets up the entry point
RUN ["chmod", "+x", "/root/entry.sh"]
ENTRYPOINT ["/root/entry.sh"]