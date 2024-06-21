#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install necessary libraries and development tools
echo "Installing necessary libraries and development tools..."
sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev swig build-essential

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools

# Install project dependencies from requirements.txt
echo "Installing project dependencies..."
pip install -r /mount/src/stocktest/requirements.txt

echo "Setup complete."
