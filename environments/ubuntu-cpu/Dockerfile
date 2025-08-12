FROM ubuntu:22.04

# -----------------------------
# Set up environment variables
# -----------------------------
ENV CONDA_DIR=/opt/miniconda
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive
ENV GRPC_INSTALL_DIR="/usr/local"

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      curl \
      vim  \
      build-essential \
      autoconf \
      libtool \
      pkg-config \
      libssl-dev \
      ca-certificates \
      swig \
      git \
      libomp5 \
      libomp-dev \
      graphviz \
    && rm -rf /var/lib/apt/lists/*


# Install CMake 3.24.2
RUN wget -qO /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-linux-x86_64.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

# -----------------------------
# Install Miniconda
# -----------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda update -n base -c defaults conda

# Copy in your conda environment YAML
COPY conda.yaml /tmp/conda.yaml

# Create quake-env
RUN conda env create -f /tmp/conda.yaml && conda clean -afy

# Activates quake-env at runtime beginning
RUN mkdir -p /working_dir
RUN conda init bash
RUN echo "conda activate quake-env" >> ~/.bashrc

WORKDIR /working_dir
CMD ["/bin/bash"]