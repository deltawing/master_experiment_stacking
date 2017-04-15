# Use wavelet transform combined with stacking algorithm to predict protein-protein interactions
Pinsan Xu, Jun Luo, Tongyi Dou*
* Corresponding author:douty@dlut.edu.cn
Aim at predict protein-protein interactions using protein's primary sequence.

## Requirements
Experimental pipeline is implemented in Python 3.
You have to import some Python 3 packages, as follows:
* numpy
* scipy
* pandas
* scikit-learn
* mlxtend
* pywt

## Project structure
The *experiment* flod contains subdirectories corresponding to different experimentsal. Each experimental directory contains a *test.py* file and necessary input files, intermediate files as well as results files. 

## Running the experiment
If all the dependencies are satisfied, you can run full experimental pipeline with:
> python test.py
