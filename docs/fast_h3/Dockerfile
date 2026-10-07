FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

# The image build sees no GPU, so the native kernels compile for the named
# architectures: Hopper, data-center Blackwell and RTX PRO 6000 Blackwell.
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" \
    UV_PROJECT_ENVIRONMENT=/opt/uniserve-venv \
    MAX_JOBS=2 \
    PATH=/opt/uniserve-venv/bin:/root/.cargo/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv build-essential pkg-config cmake \
    curl git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/uniserve
COPY . .
RUN uv sync --locked --python /usr/bin/python3.12 --extra gpu

EXPOSE 8000
ENTRYPOINT ["uniserve"]
CMD ["serve", "FastVideo/FastVideo-FastH3-8-Step-V2", "--workers", "configs/fast_h3/ulysses4.json", "--served-model-name", "FastH3", "--host", "0.0.0.0", "--max-running-requests", "2"]
