# PiNDiff-manufacturing
## Physics-integrated neural differentiable model for composites manufacturing -> [link](https://www.sciencedirect.com/science/article/pii/S0045782523000257)



This repository contains the code and materials for our paper on a physics-integrated neural differentiable (PiNDiff) model for the manufacturing of composites. The differentiable programming is done using the PyTorch library. The PiNDiff model combines incomplete physics knowledge with available measurement data within this framework to effectively learn and generalize. The method is demonstrated in the modeling of the curing process of thick thermoset composite laminates. The PiNDiff model outperforms purely data-driven models and provides a strategy for modeling phenomena with limited physics knowledge and sparse, indirect data.

## Getting Started

The code in this repository is written in Python and requires the PyTorch library to run. Before using the code, ensure that you have a compatible version of PyTorch installed on your system.

## Contents

This repository includes:

- Code for the PiNDiff model and other black-box models
- shell scripts (in the exe directory) to run the code.

## Using the Code

Following instructions can be used to run the code. 

### 1. generate data
To generate training data: ./exe/generate_traning_data.sh \
To generate testing data: ./exe/generate_testing_data.sh

### 2. train model
To train the PiNDiff model: ./exe/train_hybrid_model.sh 

### 3. test model
To test the PiNDiff model: ./exe/test_hybrid_model.sh 

## Directory
- exe: contains shell script to run the code
- output: contains all the plots and model parameters
- input: contains yaml files to change input conditions
- solver and utils: contains supporting code


## Contact

Open an issue on the Github repository if you have any questions/concerns.

## Acknowledgments

We would like to acknowledge the contributions of [Any contributors or funding sources here]. This work was supported by [Any funding sources or grants here].

## Citation

Find this useful or like this work? Cite us with:

```
@article{AKHARE2023115902,
title = {Physics-integrated neural differentiable (PiNDiff) model for composites manufacturing},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {406},
pages = {115902},
year = {2023},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2023.115902},
url = {https://www.sciencedirect.com/science/article/pii/S0045782523000257},
author = {Deepak Akhare and Tengfei Luo and Jian-Xun Wang},
keywords = {Composite cure, Differentiable programming, Neural networks, Scientific machine learning, Surrogate modeling, Operator learning}
}
```
