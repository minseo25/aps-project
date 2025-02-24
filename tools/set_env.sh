#!/bin/bash

# Start timer
start=$(date +%s)

# Python venv setup
python3 -m venv --without-pip .venv
source .venv/bin/activate

# Install pip manually
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# Install Packages (takes ~10 minutes)
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install torchtext==0.18.0 datasets numpy

# End timer
end=$(date +%s)

# Print time in minutes and seconds
minutes=$(( (end-start) / 60 ))
seconds=$(( (end-start) % 60 ))
echo "Time taken: $minutes minutes and $seconds seconds"