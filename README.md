# Quantum Machine Learning
This repository is dedicated to my capstone project completed in Fall 2023 for my Master of Science in Data Science degree (graduated December 2023) through Indiana University. Under the mentorship of Amr Sabry, I completed independent study researching & exploring the intersection of quantum computing and artificial intelligence.

In order to run locally any of the notebooks in this repository, refer to the [Creating your local environment](#Creating-your-local-environment) section below.

## Directory Structure
### Qiskit


The `qiskit` directory contains 5 folders, one for each IBM Quantum Learning course, which all include the Jupyter notebooks for each course section with example code and comments. The courses covered are listed below:

* <a href="https://qiskit.org/learn/course/introduction-course ">Introduction</a>
* <a href="https://learning.quantum.ibm.com/course/basics-of-quantum-information">Understanding Quantum Information & Computation: Basics of Quantum Information</a>
* <a href="https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms">Understanding Quantum Information & Computation: Fundamentals of Quantum Algorithms</a>
* <a href="https://learning.quantum.ibm.com/course/variational-algorithm-design">Variational Algorithm Design</a>
* <a href="https://github.com/Qiskit/textbook/tree/main/notebooks/quantum-machine-learning#">Quantum Machine Learning</a>

### Project
As part of my capstone, my project focused on quantum kernels and extending the work presented by [1]. The authors' main goal was to develop a way to screen candidate feature maps using classical computers in order to find the best one to use on quantum hardware. This was done by classically computing the quantum data states in the feature space and computing a minimum accuracy score by iterating through each of the 16 dimensions (for 2 qubit use case) and finding the best hyperplane to separate the data on a single axis. This represents a worst-case scenario by limiting the hyperplane to just one axis but gives a lower bound on the expected training accuracy that a SVC could achieve on the dataset given that particular feature map. The authors admit that this calculation works for 2 qubits but is not feasible for more qubits becuase the dimensions grow exponentially. Thus, this project looks at several different methods to compute a minimum accuracy more efficiently than the brute force calculation of looking through all dimensions and all possible hyperplanes.

### Final Paper
The final write up for my capstone can be found here: [An Intuitive Guide to Quantum Machine Learning for the Classical Data Scientist](final_paper/final_paper.pdf). The paper first provides some background information on quantum computing and the field of quantum machine learning (QML). Then it takes a special focus on quantum kernels, building up the argument as to why they're crucial to study for understanding the advantages and success of QML. Next, the paper analyzes the quantum feature space and draws intuitive parallels to classical machine learning, proving that many of the same concepts apply. Lastly, the project work of extending the work from [1] is discussed and reviewed.

## Creating your local environment

All Jupyter notebooks in this repository should be able to run using the virtual environment defined by the poetry lock and pyproject toml files at the root of this project directory. Instructions for setting up the environment can be found below. But first ensure that you have <a  href="https://git-scm.com/downloads">git</a>, <a href="https://python-poetry.org/docs/"> poetry</a> and <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">conda/miniconda</a> installed locally on your computer.


1. Clone this repository to your local computer: `git clone git@github.com:amandaturney/qml.git`. If you do not have permission, please request access.

2. Create a virtual environment by running the following command: `conda create --name qml python=3.9`

2. Activate the virtual environment: `conda activate qml`

3. Once the virtual environment is activated, poetry will "see" this environment and use it for any package installations. Then run the following command: `poetry install`

The above steps will install all the packages defined in the poetry lock file to build a consistent environment wherever this project is run.

## References

1. https://arxiv.org/pdf/1906.10467.pdf

