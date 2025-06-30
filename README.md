# Flow matching based speech enhancement
This repository contains the official PyTorch implementations for the 2025 paper:

* FlowSE: Flow Matching-based Speech Enhancement [1]



This repository builds upon a previous work, that can be found here https://github.com/sp-uhh/sgmse  

<!-- Continuous Normalizing Flow (CNF) is a method transforming a simple distribution $p(x)$ to a complex distribution $q(x)$.  

CNF is described by Oridinary Differential Equations (ODEs):  

$$ \frac{d \phi_t(x_0)}{dt} = v(t,\phi_t(x_0)), \phi_0(x_0)=x_0, x_0\sim p(\cdot) $$  

In the above ODE, a function $\phi_t$ called flow is desired such that the stochastic process $x_t=\phi_t(x_0)$ has a marginal distribution $p_t(\cdot)$ such that $p_1(\cdot ) = q(\cdot)$.   
In the above equation, although the condition that $\phi_0(x_0)$ follows $p$ is imposed (inital value problem), by chain rule replacing $t$ with $1-t$, CNF is can be desribed as:  

$$\frac{d\phi_t(x_1)}{dt} = v_t(t,\phi_t(x_1)), \phi_1(x_1)=x_1, x_1 \sim p(\cdot)$$  

It means that it does not matter that the simpled distribution is located at which time point.
Demo page: https://seongqjini.com/speech-enhancement-with-flow-matching-method/ -->


## Citations / References
[1] Seonggyu Lee, Sein Cheong, Sangwook Han, Jong Won Shin. 
[*FlowSE: Flow Matching-based Speech Enhancement*](https://ieeexplore.ieee.org/document/10888274), ICASSP, 2025.


