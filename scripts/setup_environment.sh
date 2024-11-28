#!/bin/bash

set -e

ENVIRONMENT_NAME="myenv"


echo "Creating a virtual environment..."
python -m venv $ENVIRONMENT_NAME


echo "Activating the virtual environment..."
$ENVIRONMENT_NAME/Scripts/activate


echo "pip udating..."
pip install --upgrade pip


echo "Loading required libraries..."
pip install numpy matplotlib keras tensorflow 


echo "Creating requirements.txt file..."
pip freeze > requirements.txt


echo "The installation is complete. You can use the 'deactivate' command to close it."
