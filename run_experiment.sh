#!/bin/bash

# This script runs the experiment for the given model and dataset.

# Execute benchmark for PETS dataset.
python -m main benchmark pets

# Execute benchmark for VOC dataset.
python -m main benchmark voc