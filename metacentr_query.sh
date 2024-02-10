#!/bin/bash

# Navigate to the specified directory
cd /storage/brno2/home/zovi

# Load the required Python module
module add python/3.9.12-gcc-10.2.1-rg2lpmk

# Activate the Python virtual environment
source lvd_e/bin/activate

# Navigate to the script's directory
cd lvd/lmi_examples/

# Execute the Python script
python query_hm.py
