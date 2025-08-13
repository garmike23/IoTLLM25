#!/bin/bash

set -e  # Exit on any error

cd ~/Desktop/IoTLLM25/

# System dependencies
sudo apt-get update
sudo apt-get install -y python3-virtualenv python3-virtualenvwrapper python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

# Add virtualenvwrapper settings to .bashrc if not already present
if ! grep -Fxq "export WORKON_HOME=\$HOME/.virtualenvs" ~/.bashrc; then
    echo "export WORKON_HOME=\$HOME/.virtualenvs" >> ~/.bashrc
    echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
fi

# Source virtualenvwrapper for this script's session
export WORKON_HOME=$HOME/.virtualenvs
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh

# Create and activate virtualenv
if [ ! -d "$WORKON_HOME/venv" ]; then
    mkvirtualenv venv
else
    workon venv
fi

# Install all dependencies
if [ -f requirements.txt ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

echo "All dependencies installed successfully in the 'venv' virtual environment."
