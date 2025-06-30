# Flow matching based speech enhancement
This repository contains the official PyTorch implementations for the 2025 paper:

* FlowSE: Flow Matching-based Speech Enhancement [1]


This repository builds upon previous great works:
* [SGMSE] https://github.com/sp-uhh/sgmse  
* [SGMSE-CRP] https://github.com/sp-uhh/sgmse_crp
* [BBED]  https://github.com/sp-uhh/sgmse-bbed


## Evaluation
* To evaluate on a test set, run
  ```bash
  python enhancement.py --test_dir <your_test_dataset_dir> --folder_destination <your_enh_result_save_dir> --ckpt <path_to_model_checkpoint> --N <num_of_time_steps>
  ```

## Citations / References
[1] Seonggyu Lee, Sein Cheong, Sangwook Han, Jong Won Shin. 
[*FlowSE: Flow Matching-based Speech Enhancement*](https://ieeexplore.ieee.org/document/10888274), ICASSP, 2025.

```
@INPROCEEDINGS{10888274,
  author={Lee, Seonggyu and Cheong, Sein and Han, Sangwook and Shin, Jong Won},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={FlowSE: Flow Matching-based Speech Enhancement}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Perturbation methods;Computational modeling;Diffusion processes;Speech enhancement;Signal processing;Diffusion models;Vectors;Noise measurement;Kernel;Tuning;speech enhancement;generative model;flow matching;diffusion model},
  doi={10.1109/ICASSP49660.2025.10888274}}

```

이거 왜 안되

<!-- Continuous Normalizing Flow (CNF) is a method transforming a simple distribution $p(x)$ to a complex distribution $q(x)$.  

CNF is described by Oridinary Differential Equations (ODEs):  

$$ \frac{d \phi_t(x_0)}{dt} = v(t,\phi_t(x_0)), \phi_0(x_0)=x_0, x_0\sim p(\cdot) $$  

In the above ODE, a function $\phi_t$ called flow is desired such that the stochastic process $x_t=\phi_t(x_0)$ has a marginal distribution $p_t(\cdot)$ such that $p_1(\cdot ) = q(\cdot)$.   
In the above equation, although the condition that $\phi_0(x_0)$ follows $p$ is imposed (inital value problem), by chain rule replacing $t$ with $1-t$, CNF is can be desribed as:  

$$\frac{d\phi_t(x_1)}{dt} = v_t(t,\phi_t(x_1)), \phi_1(x_1)=x_1, x_1 \sim p(\cdot)$$  

It means that it does not matter that the simpled distribution is located at which time point.
Demo page: https://seongqjini.com/speech-enhancement-with-flow-matching-method/ -->


