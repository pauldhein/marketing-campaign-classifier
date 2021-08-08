FROM  ubuntu:20.04
CMD   bash

# ==============================================================================
# INSTALL SOFTWARE VIA THE UBUNTU PACKAGE MANAGER
# =============================================================================
RUN apt-get update && \
    apt-get -y --no-install-recommends install \
    python3-dev python3-pip python3-venv
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# =============================================================================

# =============================================================================
# CREATE A PYTHON VENV AND UPGRADE PYTHON TOOLS
# =============================================================================
ENV VIRTUAL_ENV=/opt/telemarket_venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade setuptools
RUN pip install wheel
# =============================================================================

# =============================================================================
# INSTALL REQUIRED PYTHON PACKAGES AND SETUP THE SOURCE ENVIRONMENT
# =============================================================================
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /data
COPY data /data/

RUN mkdir -p /telemarket_model
COPY source /telemarket_model/
WORKDIR /telemarket_model
# =============================================================================