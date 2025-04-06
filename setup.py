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
    python_requires =   ">=3.11",
    install_requires =  [
                            "donfig>=0.8",
                            "GitPython",
                            "nibabel",
                            "numcodecs[crc32c]>=0.14",
                            "numpy>=1.25",
                            "packaging>=22.0",
                            "rich",
                            "segment-anything-py",
                            "segmentation-models-pytorch",
                            "torch",
                            "typing_extensions>=4.9",
                            "zarr"
                        ]
)