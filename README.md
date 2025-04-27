# Segmentation Model Benchmarking

This script conducts benchmarking experiments for various segmentation models.

## Models tested:
* DeepLab-V3+   (https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus)
* FPN           (https://smp.readthedocs.io/en/latest/models.html#fpn)
* Seg-Former    (https://smp.readthedocs.io/en/latest/models.html#segformer)
* U-Net         (https://smp.readthedocs.io/en/latest/models.html#unet)

## Datasets used:
* Oxford IIIT Pets          (https://www.robots.ox.ac.uk/~vgg/data/pets/)
* Pascal VOC Segmentation   (http://host.robots.ox.ac.uk/pascal/VOC/)

## For each model:dataset experiment, the following metrics will be reported in a "results" directory:
* F1 Score
* Dice Score
* Precision
* Recall
* Computational Costs

## Running Experiments:
Execute full suite of experiments by running: `bash ./run_experiments.sh`

## Contributions: 
All implementation and consulting of code fulfilled evenly by both team members.
* Team Member 1 (Azwaad Labiba Mohiuddin):
  * Datasets
  * Metrics
  * Data Visualization
* Team Member 2 (Gabriel C. Trahan):
  * Models
  * Benchmarking Process
  * Data Preprocessing

## Authors:
* Gabriel C. Trahan         (C00058009)
* Azwaad Labiba Mohiuddin   (C00580385)
