#!/bin/bash
# Downloads MASSIVE 1.1 (Amazon Science, CC BY 4.0) and the multilingual MiniLM encoder.
set -eu
cd "$(dirname "$0")"
mkdir -p data models
if [ ! -d data/1.1 ]; then
  curl -L -o data/massive.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
  tar -xzf data/massive.tar.gz -C data
fi
if [ ! -f models/mMiniLM/model.safetensors ]; then
  python -c "from huggingface_hub import snapshot_download as d; d('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', local_dir='models/mMiniLM')"
fi
