# Quantum Machine Learning
This repository is dedicated to the DSCI-D699 course of independent study for researching & exploring the intersection of quantum computing and artificial intelligence.

## Creating your local environment

All Jupyter notebooks in this repository should be able to run using the virtual environment defined by the poetry lock and pyproject toml files at the root of this project directory. Instructions for setting up the environment can be found below. But first ensure that you have <a  href="https://git-scm.com/downloads">git</a>, <a href="https://python-poetry.org/docs/"> poetry</a> and <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">conda/miniconda</a> installed locally on your computer.


1. Clone this repository to your local computer: `git clone git@github.com:amandaturney/qml.git`. If you do not have permission, please request access.

2. Create a virtual environment by running the following command: `conda create --name qml python=3.9`

2. Activate the virtual environment: `conda activate qml`

3. Once the virtual environment is activated, poetry will "see" this environment and use it for any package installations. Then run the following command: `poetry install`

The above steps will install all the packages defined in the poetry lock file to build a consistent environment wherever this project is run.

