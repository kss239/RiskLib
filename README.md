# RiskLib

RiskLib is currently a work in progress barebones Julia package that currently provides portfolio optimization techniques using variaous risk measures for financial asset allocation.  The package contains two main files: measures.jl and RiskLib.jl.

## Describe

### measures.jl

This file contains functions for calculating different risk measures. These measures can be grouped into three categories:

1. Dispersion Risk Measures
2. Downside Risk Measures
3. Drawdown Risk Measures

Examples of risk measures in this file:
* Maximum Drawdown
* Standard Deviation
* Mean Absolute Deviation
* Value-at-Risk (VaR)
* Conditional Value-at-Risk (CVaR)

### RiskLib.jl

This file contains functions for monte carlo portfolio optimization using various techniques, including:

1. Mean-Risk Optimization
2. Risk Parity Optimization

Each of these techniques requires an obj_type parameter, which specifies the optimization objective:

- "min_risk": Minimize the risk of the portfolio
- "max_return": Maximize the expected return of the portfolio
- "max_utility": Maximize the expected utility (expected return minus a penalty for risk)
- "max_ratio": Maximize the ratio of expected return to risk

## Getting Started

The package provides two portfolio optimization functions:

1. mean_risk_optimization_montecarlo
2. risk_parity_optimization_montecarlo

To use these functions, first import the package:
~~~
push!(LOAD_PATH,"src/RiskLib.jl") # replace "src/RiskLib.jl" with the location of RiskLib
using RiskLib
~~~
and have a dataset ready, the example data set below is a 5 asset x 10yrs of annual returns
~~~
returns = [
    0.05  0.12  0.08  0.02  0.10;
    0.03  0.10  0.05  0.01  0.11;
    0.07  0.14  0.10  0.03  0.09;
    0.02  0.08  0.06  0.00  0.12;
    0.04  0.09  0.07  0.02  0.10;
    0.06  0.11  0.09  0.04  0.08;
    0.01  0.07  0.04  0.00  0.13;
    0.03  0.08  0.05  0.02  0.11;
    0.05  0.10  0.07  0.03  0.09;
    0.04  0.09  0.06  0.01  0.10
]
~~~
begin to call functions from the package and specify for each function each parameter
~~~
mean_risk_optimization_montecarlo(returns, value_at_risk, "max_utility", lambda=0.5)
~~~
this will return the found asset weights that fulfill the objective function use of the risk metric.

*Note: Not all risk_measures, obj_types or portfolio optimization functions are interchangeable*
