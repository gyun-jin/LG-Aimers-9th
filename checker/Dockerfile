# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11.15

ENV PATH="/opt/python/bin:${PATH}" \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Seoul

# Keep the documented evaluator system packages in the final image. Extra
# libraries below are required to build CPython with its standard modules.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libffi-dev libblas3 liblapack3 \
        libomp-dev tzdata unzip p7zip-full gfortran libatlas-base-dev \
        default-jre-headless cmake pkg-config ninja-build libgl1 libglib2.0-0 \
        ca-certificates curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev liblzma-dev libncursesw5-dev libgdbm-dev libnss3-dev \
        uuid-dev tk-dev xz-utils \
    && curl -fsSLo /tmp/Python.tar.xz \
        "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
    && mkdir /tmp/Python \
    && tar -xJf /tmp/Python.tar.xz -C /tmp/Python --strip-components=1 \
    && cd /tmp/Python \
    && ./configure --prefix=/opt/python --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make install \
    && ln -s /opt/python/bin/python3.11 /opt/python/bin/python \
    && ln -s /opt/python/bin/pip3.11 /opt/python/bin/pip \
    && rm -rf /tmp/Python /tmp/Python.tar.xz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY checker/base-requirements.txt /opt/checker/base-requirements.txt

# PyTorch publishes CUDA 12.8 wheels on its own index. Install the remaining
# evaluator packages from PyPI, then allow the submission requirements to
# replace base versions just as the evaluation service does.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.7.1 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r /opt/checker/base-requirements.txt

ARG SUBMISSION_ZIP=baseline_submit.zip
RUN --mount=type=bind,source=${SUBMISSION_ZIP},target=/tmp/submission.zip,readonly \
    unzip -p /tmp/submission.zip requirements.txt > /tmp/submission-requirements.txt \
    && test -s /tmp/submission-requirements.txt \
    && python -m pip install -r /tmp/submission-requirements.txt \
    && rm /tmp/submission-requirements.txt

COPY checker.py /opt/checker/checker.py
COPY checker/container-entrypoint.sh /opt/checker/container-entrypoint.sh
RUN chmod 0755 /opt/checker/container-entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/opt/checker/container-entrypoint.sh"]
