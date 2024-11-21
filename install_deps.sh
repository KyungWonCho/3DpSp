#!/bin/bash

conda create -n 3DpSp python=3.9

git submodule update --init --recursive

conda run -n 3DpSp pip install -r requirements.txt

git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
conda run -n 3DpSp pip install .
cd ..
rm -rf nvdiffrast

git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer
sed -i 's/AT_CHECK/TORCH_CHECK/g' neural_renderer/cuda/load_textures_cuda.cpp
sed -i 's/AT_CHECK/TORCH_CHECK/g' neural_renderer/cuda/create_texture_image_cuda.cpp
sed -i 's/AT_CHECK/TORCH_CHECK/g' neural_renderer/cuda/rasterize_cuda.cpp
conda run -n 3DpSp pip install .
cd ..
rm -rf neural_renderer

cd dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r insightface/recognition/arcface_torch ./models/
rm -rf insightface
cd ../../..