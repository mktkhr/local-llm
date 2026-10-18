# PD分離 PoC 用の llama-server (CUDA)。
#
# ホストに CUDA Toolkit を入れずに、llama.cpp の特定タグを固定ビルドする。
# PD分離では Prefill 側と Decode 側で llama.cpp のバージョンを厳密に揃える
# 必要があるため、公式イメージ(commit 単位のタグが古い世代までしか無い)ではなく
# 自前ビルドする。
ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# 両マシンで一致させるタグ。変更するときは Decode 側も同時に変える。
ARG LLAMA_TAG=b10679
# RTX 4080 SUPER (AD103) は compute capability 8.9。
# 単一アーキに絞るとビルド時間が大幅に短くなる。
ARG CUDA_ARCH=89

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ca-certificates \
        libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp.git . \
    && git log -1 --format='%H' > /commit.txt

# -Wl,--allow-shlib-undefined は必須。
# libggml-cuda.so は CUDA Driver API (cuGetErrorString 等) を参照するが、
# devel イメージには stub しか無いためリンク時に未解決となる。実行時に
# ホストの本物の libcuda.so へ解決させるため、未解決シンボルを許可する。
# 公式の .devops/cuda.Dockerfile と同じ対処。
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-server -j "$(nproc)"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ggml はバックエンドを実行ファイルと同じディレクトリから dlopen するため、
# build/bin をディレクトリごと配置する。
COPY --from=build /src/build/bin/ /app/
COPY --from=build /commit.txt /app/commit.txt
ENV LD_LIBRARY_PATH=/app

ENTRYPOINT ["/app/llama-server"]
