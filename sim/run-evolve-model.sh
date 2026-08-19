#!/bin/bash
# Wrapper script for model evolution
cd "$(dirname "$0")"
python evolve-model.py --iterations 8 --ny 40 --case-name new-evolve
