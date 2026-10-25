# syntax=docker/dockerfile:1
#
# sc-dandelion container: dandelion plus IgBLAST, BLAST, Change-O, TIgGER and
# pre-built germline reference databases, for V(D)J preprocessing.
#
# Build from the REPOSITORY ROOT (dandelion is installed from the local source):
#   docker build -f container/Dockerfile -t sc-dandelion .
#
# Use with Docker:
#   docker run --rm -v "$PWD":/data -w /data --user "$(id -u):$(id -g)" \
#       sc-dandelion dandelion-preprocess --help
# Use with Apptainer/Singularity (HPC):
#   apptainer pull docker://zktuong/sc-dandelion:latest
#   apptainer run -B $PWD sc-dandelion_latest.sif dandelion-preprocess --help
#
# linux/amd64 only: IgBLAST is downloaded as an x64 Linux binary.

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Version given to the dandelion package. setuptools_scm can't see .git inside
# the build, so CI passes the version it computes from the checkout.
ARG DANDELION_VERSION=0.0.0+docker
ARG VCS_REF=unknown
# 1 = keep going if the IMGT database download fails (PR/weekly checks), and
#     treat the other databases as best effort. 0 = any failure fails the build.
ARG ALLOW_DB_FAILURE=0
# 1 = also build the OGRDB database (best effort).
ARG INCLUDE_OGRDB=0

LABEL org.opencontainers.image.title="sc-dandelion" \
      org.opencontainers.image.description="dandelion single-cell BCR/TCR (V(D)J) analysis with IgBLAST, BLAST, Change-O, TIgGER and germline reference databases" \
      org.opencontainers.image.source="https://github.com/tuonglab/dandelion" \
      org.opencontainers.image.documentation="https://sc-dandelion.readthedocs.io/" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.version="${DANDELION_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Everything the old Singularity %environment / conda activate provided, as
# plain ENV so it works without a login shell under Docker and Apptainer.
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    CONDA_DIR=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/sc-dandelion-container \
    CONDA_DEFAULT_ENV=sc-dandelion-container
ENV PATH=/share/bin:${CONDA_PREFIX}/bin:/share/ncbi-igblast/bin:${CONDA_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GERMLINE=/share/database/germlines/ \
    IGDATA=/share/database/igblast/ \
    BLASTDB=/share/database/blast/ \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# System packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git gcc language-pack-en \
    && rm -rf /var/lib/apt/lists/*

# Miniforge + the conda environment
COPY container/environment.yml /tmp/environment.yml
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && mamba env create -y -n sc-dandelion-container -f /tmp/environment.yml \
    && mamba clean -afy \
    && rm /tmp/environment.yml

# IgBLAST (latest NCBI release) at a fixed path
RUN set -eu; \
    base="https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST"; \
    version=$(curl -fsSL "${base}/" | grep -oE 'igblast-[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d- -f2); \
    echo "IgBLAST ${version}"; \
    curl -fsSL -o /tmp/igblast.tar.gz "${base}/ncbi-igblast-${version}-x64-linux.tar.gz"; \
    mkdir -p /share; \
    tar -xzf /tmp/igblast.tar.gz -C /share; \
    mv "/share/ncbi-igblast-${version}" /share/ncbi-igblast; \
    rm /tmp/igblast.tar.gz; \
    echo "${version}" > /share/IGBLAST_VERSION

# Reference databases
COPY container/scripts/utils.py \
     container/scripts/prepare_imgt_database.py \
     container/scripts/prepare_kiarva_database.py \
     container/scripts/prepare_gkhlab_database.py \
     container/scripts/prepare_ogrdb_database.py \
     /share/
RUN set -u; \
    cd /share; \
    echo ">>> Preparing IMGT database"; \
    if ! python /share/prepare_imgt_database.py \
            --outdir /share/database --igblast_dir /share/ncbi-igblast; then \
        if [ "${ALLOW_DB_FAILURE}" = "1" ]; then \
            echo "WARNING: IMGT database preparation failed; container tests will be skipped."; \
            touch /share/DATABASE_FAILED; \
        else \
            echo "ERROR: IMGT database preparation failed."; exit 1; \
        fi; \
    fi; \
    for db in kiarva gkhlab; do \
        echo ">>> Preparing ${db} database"; \
        if ! python "/share/prepare_${db}_database.py" --outdir /share/database; then \
            if [ "${ALLOW_DB_FAILURE}" = "1" ]; then \
                echo "WARNING: ${db} database preparation failed; continuing."; \
            else \
                echo "ERROR: ${db} database preparation failed."; exit 1; \
            fi; \
        fi; \
    done; \
    if [ "${INCLUDE_OGRDB}" = "1" ]; then \
        echo ">>> Preparing OGRDB database (best effort)"; \
        python /share/prepare_ogrdb_database.py --outdir /share/database \
            || echo "WARNING: OGRDB database preparation failed; continuing."; \
    fi

# dandelion itself, from this checkout
COPY pyproject.toml README.md LICENSE MANIFEST.in /tmp/dandelion-src/
COPY src /tmp/dandelion-src/src
RUN SETUPTOOLS_SCM_PRETEND_VERSION="${DANDELION_VERSION}" \
        python -m pip install --no-cache-dir /tmp/dandelion-src \
    && rm -rf /tmp/dandelion-src \
    && python -c "import dandelion; print('dandelion', dandelion.__version__)"

# Command-line entry points: `dandelion-preprocess` and `changeo-clonotypes`
COPY container/scripts/dandelion_preprocess.py \
     container/scripts/changeo_clonotypes.py \
     /share/
RUN mkdir -p /share/bin \
    && printf '#!/bin/sh\nexec python /share/dandelion_preprocess.py "$@"\n' > /share/bin/dandelion-preprocess \
    && printf '#!/bin/sh\nexec python /share/changeo_clonotypes.py "$@"\n' > /share/bin/changeo-clonotypes \
    && chmod 755 /share/bin/* /share/*.py

# Container self-tests (run in CI with: pytest -p no:cacheprovider /tests)
COPY container/tests /tests

# Apptainer runs the image read-only as the calling user, so everything must
# be world-readable.
RUN chmod -R a+rX /share /tests "${CONDA_DIR}"

# No ENTRYPOINT: `docker run IMAGE <cmd>` and `apptainer run IMAGE <cmd>` both
# run <cmd> directly; with no arguments you get a shell.
CMD ["/bin/bash"]
