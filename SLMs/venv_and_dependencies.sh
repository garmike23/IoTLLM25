#!/bin/bash
set -e

echo "===> Updating system packages"
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

echo "===> Creating virtual environment called venv"
python3 -m venv venv

echo "===> Activating virtual environment"
source venv/bin/activate

echo "===> Upgrading pip"
pip install --upgrade pip

echo "===> Installing training dependencies"
pip install torch transformers datasets matplotlib

echo "===> Installing ONNX export and quantization dependencies"
pip install onnx onnxruntime onnxruntime-tools optimum

echo "===> Installation complete"
echo "To activate your environment later, run: source venv/bin/activate and to deactivate run: deactivate"
