from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mudes",
    version="0.1.0",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="Toxic Spans Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharinduDR/MUDES",
    packages=find_packages(exclude=("examples",)),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "spacy == 2.3.4",
        "pandas == 1.1.4",
        "transformers == 3.5.1",
        "seqeval == 1.2.2",
        "tensorboardX == 2.1",
        "wandb == 0.10.11",
        "googledrivedownloader == 0.4"
    ],
)
