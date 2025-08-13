#!/bin/bash

set -e

cd ~/Desktop/IoTLLM25/SLMs/DistilGPT2/

python3 local_download.py

python3 train_gpt_model.py

python3 optimize.py

python3 test_gpt_model.py

python3 test_baseline_to_csv.py

python3 test_onnx_to_csv.py

python3 test_onnx_and_int8_to_csv.py

python3 graph_csv_results.py
