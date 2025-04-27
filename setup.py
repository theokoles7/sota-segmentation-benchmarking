"""Setup utility."""

from setuptools import find_packages, setup

setup(
    name =              "segment",
    version =           "0.0.1",
    author =            (
                            "Gabriel C. Trahan, "
                            "Azwaad Labiba Mohiuddin"
                        ),
    author_email =      (
                            "gabriel.trahan1@louisiana.edu, "
                            "azwaad-labiba.mohiuddin1@louisiana.edu"
                        ),
    description =       (
                            "CSCE-508 (Image Processing) project focues on benchmarking the "
                            "state-of-the-art segmentation models (as of 2025)."
                        ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/sota-segmentation-benchmarking",
    packages =          find_packages(),
    python_requires =   ">=3.11",
    install_requires =  [
                            "matplotlib",
                            "medpy",
                            "numpy",
                            "pandas",
                            "segmentation_models_pytorch",
                            "thop",
                            "torch",
                            "tqdm"
                        ]
)