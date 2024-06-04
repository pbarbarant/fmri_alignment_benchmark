#!/bin/bash

benchopt clean
benchopt run --slurm slurm_config.yaml -d IBC_RSVPLanguage -s identity -s fugw
python python benchmark_utils/generate_figures.py