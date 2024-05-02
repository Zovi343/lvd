#!/bin/bash
# LVD MODIFICATION START
cd /storage/brno2/home/zovi

module add python/3.9.12-gcc-10.2.1-rg2lpmk

source lvd_e/bin/activate

cd lvd/lmi_examples/

python query_hm.py
# LVD MODIFICATION END