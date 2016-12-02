# Adversarial Multiclass Classification: A Risk Minimization Perspective (NIPS 2016)
This repository is a code example of a the paper: 
[Adversarial Multiclass Classification: A Risk Minimization Perspective](https://papers.nips.cc/paper/6088-adversarial-multiclass-classification-a-risk-minimization-perspective)

Full paper: [https://www.cs.uic.edu/~rfathony/pdf/fathony2016adversarial.pdf](https://www.cs.uic.edu/~rfathony/pdf/fathony2016adversarial.pdf)

### Abstract

Recently proposed adversarial classification methods have shown promising results for cost sensitive and multivariate losses. In contrast with empirical risk minimization (ERM) methods, which use convex surrogate losses to approximate the desired non-convex target loss function, adversarial methods minimize non-convex losses by treating the properties of the training data as being uncertain and worst case within a minimax game. Despite this difference in formulation, we recast adversarial classification under zero-one loss as an ERM method with a novel prescribed loss function. We demonstrate a number of theoretical and practical advantages over the very closely related hinge loss ERM methods. This establishes adversarial classification under the zero-one loss as a method that fills the long standing gap in multiclass hinge loss classification, simultaneously guaranteeing Fisher consistency and universal consistency, while also providing dual parameter sparsity and high accuracy predictions in practice.


# Setup

The source code is written in [Julia](http://julialang.org/) version 0.5.0.

### Dependency
The code depends on the followong Julia Packages:

1. [Optim.jl](https://github.com/JuliaOpt/Optim.jl)
2. [Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl)
3. [Mosek.jl](https://github.com/JuliaOpt/Mosek.jl)

[Optim.jl](https://github.com/JuliaOpt/Optim.jl) is used in the primal BFGS optimization. 
To run the dual constraint generation algorithm, a Quadratic Programming solver ([Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl) 
or [Mosek.jl](https://github.com/JuliaOpt/Mosek.jl)) is required. Please refer to each package's instruction for the installation.

### Example

Three example files are provided: 

* `example.jl` :
run dual constraint generation algorithm for training. 

* `example_kernel.jl` :
run dual constraint generation algorithm for training with Gaussian kernel.

* `example_primal.jl`: 
run primal optimization algorithm (BFGS or SGD) for training.

In each file, the code will run training with k-fold cross validation for the example dataset (`glass`). 
After finding the best setting, it will run testing phase.

To change the training settings, please directly edit the setting values in the given example.

To run the code, execute (in terminal):
```
julia example.jl
```

# Citation (BibTeX)
```
@incollection{fathony2016adversarial,
title = {Adversarial Multiclass Classification: A Risk Minimization Perspective},
author = {Fathony, Rizal and Liu, Anqi and Asif, Kaiser and Ziebart, Brian},
booktitle = {Advances in Neural Information Processing Systems 29},
pages = {559--567},
year = {2016},
}
```
# Acknowledgements 
This research was supported as part of the Future of Life Institute (futureoflife.org) FLI-RFP-AI1 program, grant\#2016-158710 and by NSF grant RI-\#1526379.
