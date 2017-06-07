# keras_gpyopt
Using Bayesian Optimization to optimize hyper parameter in Keras-made neural network model.
This repository is a sample code for running Keras neural network model for MNIST, tuning hyper parameter with Bayesian Optimization.

# Installing GpyOpt
Bayesian Optimization in the program is run by GpyOpt library. This is a Python library for Bayesian Optimization.
http://sheffieldml.github.io/GPyOpt/
For installation of GpyOpt, run the followin commands.

```
conda update scipy
pip install GPy
pip install gpyopt
```

# Bayesian Optimization
Bayesian Optimization assumes the equation between input and output as black box and tries to acquire distribution of the output by exploring and observing various inputs and outputs.
Bayesian Optimization improves distribution assumption by sampling the inputs and outputs to get close to the actual distribution in exploitable time.
See below for more.
http://www.mlss2014.com/files/defreitas_slides1.pdf


# How to use
Git clone the repository and run bopt_nn.py.

```
https://github.com/shibuiwilliam/keras_gpyopt.git
cd keras_gpyopt
python bopt_nn.py
```

This will give you optimized parameter for MNIST classification on Keras as well as its accuracy.

