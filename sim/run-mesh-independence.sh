#!/bin/bash
# Wrapper script for mesh independence study
cd "$(dirname "$0")"
python run.py --ny "$1" -f --turbulence-model "$2"
