FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ARG UID=1000
ARG GID=1000
ARG REPOSITORY_NAME
ARG UNAME=$UNAME
ARG REQUIREMENTS_PATH="requirements.txt"
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/home/$UNAME/.local/bin:$PATH"

# setup user
RUN groupadd -g $GID -o $UNAME
RUN useradd -u $UID -g $GID -o --create-home $UNAME
WORKDIR /home/$UNAME
RUN chown -R $UNAME:$UNAME /home/$UNAME

# install Python
RUN apt-get update -y && \
    apt-get install -y software-properties-common build-essential curl make vim && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

USER $UNAME

# install python packages
# note: triton is uninstalled because it has really weird version dependencies and backwards incompatibility
COPY --chown=$UNAME "$REQUIREMENTS_PATH" /tmp/requirements.txt
RUN python3.11 -m pip install --no-cache-dir --upgrade setuptools distlib pip && \
    python3.11 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python3.11 -m pip uninstall -y triton && \
    rm /tmp/requirements.txt

# copy files
COPY --chown=$UNAME . $REPOSITORY_NAME

WORKDIR /home/$UNAME/$REPOSITORY_NAME

ENTRYPOINT ["/bin/bash"]
