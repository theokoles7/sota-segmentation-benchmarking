"""Setup utility."""

from setuptools import find_packages, setup

setup(
    name =              "segment",
    version =           "0.0.1",
    author =            (
                            "Azwaad Labiba Mohiuddin ,"
                            "Gabriel C. Trahan"
                        ),
    author_email =      (
                            "azwaad-labiba.mohiuddin1@louisiana.edu, "
                            "gabriel.trahan1@louisiana.edu"
                        ),
    description =       (
                            "CSCE-508 (Image Processing) project focues on benchmarking the "
                            "state-of-the-art segmentation models (as of 2025)."
                        ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/sota-segmentation-benchmarking",
    packages =          find_packages(),
    python_requires =   ">=3.10",
    install_requires =  [

                        ]
)