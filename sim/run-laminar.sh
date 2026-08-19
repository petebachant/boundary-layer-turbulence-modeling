#!/bin/bash
# Wrapper script for laminar simulation
cd "$(dirname "$0")"
python run.py -f --turbulence-model laminar --ny 40
