# Qiskit Courses

This subdirectory contains all the Jupyter notebooks for each of the 5 Qiskit courses listed below:

* <a href="https://qiskit.org/learn/course/introduction-course ">Introduction</a>
* <a href="https://learning.quantum.ibm.com/course/basics-of-quantum-information">Understanding Quantum Information & Computation: Basics of Quantum Information</a>
* <a href="https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms">Understanding Quantum Information & Computation: Fundamentals of Quantum Algorithms</a>
* <a href="https://learning.quantum.ibm.com/course/variational-algorithm-design">Variational Algorithm Design</a>
* <a href="https://github.com/Qiskit/textbook/tree/main/notebooks/quantum-machine-learning#">Quantum Machine Learning</a>

All of the notebooks can be ran using the virtual environment defined by the poetry lock and pyproject toml file defined in the parent directory. Instructions to set that up are found in the README file at the root of this project directory.

For any notebook cells that output a circuit drawing, your local set up may use the default text backend to represent the circuit. If you would like to output the visualizations as seen in the qiskit text book, you can create a `settings.conf` file (usually found in ~/.qiskit/) with the contents:

> [default]<br>
> circuit_drawer = mpl

which will make the default drawer backend be MatPlotLib.