# Numerical differentiation 

Calculation of the derivative of a function at a point using a finite difference 
formula of the second order of accuracy on two grids.

Here is an example of the result of the program.

![alt text](https://github.com/SergeiSd/numerical-methods/blob/main/Numerical%20differentiation/images/program_result.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing 
purposes.

## Prerequisites

![](https://img.shields.io/badge/numpy-v.1.19-inactivegreen) ![](https://img.shields.io/badge/sympy-v.1.8-inactivegreen) ![](https://img.shields.io/badge/prettytable-v.2.1-inactivegreen)

## Installing

Just git clone this repo and you are good to go.

    git clone https://github.com/SergeiSd/numerical-methods.git
    
## Launching the program
    
    # From the repo's root directory
    python3 numerical_differentiation.py \
        --function='sin(0.5x)' \
        --point=pi/2 \
        --step=1 \
        --r_value=0.5
        
