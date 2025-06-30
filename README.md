# Flow matching based speech enhancement
This repository contains the official PyTorch implementations for the 2025 paper:
* FlowSE: Flow Matching-based Speech Enhancement [1]



<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10888274" target="_blank">
    <img src="https://seongqjini.com/wp-content/uploads/2025/02/Flowse_ani.gif" alt="FlowSE fig1" width="600"/>
  </a>
  
</p>
<p align="center">  
   <a href="https://youtu.be/sjYstc5ss-g?si=h7CSjjvYb3BwdT2f" target="_blank">
    <img src="https://img.youtube.com/vi/sjYstc5ss-g/0.jpg" width="600" alt="YouTube English Video"/>
  </a>
  <p align="center">
  <a href="https://youtu.be/sjYstc5ss-g?si=3yEjGvfJ4RdgKfuh">Presentation video [english]</a>, <a href="https://youtu.be/PI4qyd4YDJk?si=xhrrJ-MoRSewkQ36"> Presentation video [korean] </a>
  </a>
  </p>
</p>
Speech examples are available on our [DEMOpage](https://seongqjini.com/speech-enhancement-with-flow-matching-method/).







This repository builds upon previous great works:
* [SGMSE] https://github.com/sp-uhh/sgmse  
* [SGMSE-CRP] https://github.com/sp-uhh/sgmse_crp
* [BBED]  https://github.com/sp-uhh/sgmse-bbed

## Installation
* Create a new virtual environment with Python 3.10 (we have not tested other Python versions, but they may work).
* Install the package dependencies via ```pip install -r requirements.txt```.
* [**W&B**](https://wandb.ai/) is required.


## Training
Training is done by executing train.py. A minimal running example with default settings (as in our paper [1]) can be run with

```python
python train.py --base_dir <your_dataset_dir>
```
where `your_dataset_dir` should be a containing subdirectories `train/` and `valid/` (optionally `test/` as well). 

Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To get the training set WSJ0-CHIME3, we refer to https://github.com/sp-uhh/sgmse and execute create_wsj0_chime3.py.

To see all available training options, run python train.py --help. 

## Evaluation
  To evaluate on a test set, run


  ```bash
  python enhancement.py --test_dir <your_test_dataset_dir> --folder_destination <your_enh_result_save_dir> --ckpt <path_to_model_checkpoint> --N <num_of_time_steps>
  ```

`your_test_dataset_dir` should contain a subfolder `test` which contains subdirectories `clean` and `noisy`. `clean` and `noisy` should contain .wav files.
## Citations / References
[1] Seonggyu Lee, Sein Cheong, Sangwook Han, Jong Won Shin. 
[*FlowSE: Flow Matching-based Speech Enhancement*](https://ieeexplore.ieee.org/document/10888274), ICASSP, 2025.

``` bib
@INPROCEEDINGS{10888274,
  author={Seonggyu Lee and Sein Cheong and Sangwook Han and Jong Won Shin},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={FlowSE: Flow Matching-based Speech Enhancement}, 
  year={2025},
  doi={10.1109/ICASSP49660.2025.10888274}}

```


<!-- Continuous Normalizing Flow (CNF) is a method transforming a simple distribution $p(x)$ to a complex distribution $q(x)$.  

CNF is described by Oridinary Differential Equations (ODEs):  

$$ \frac{d \phi_t(x_0)}{dt} = v(t,\phi_t(x_0)), \phi_0(x_0)=x_0, x_0\sim p(\cdot) $$  

In the above ODE, a function $\phi_t$ called flow is desired such that the stochastic process $x_t=\phi_t(x_0)$ has a marginal distribution $p_t(\cdot)$ such that $p_1(\cdot ) = q(\cdot)$.   
In the above equation, although the condition that $\phi_0(x_0)$ follows $p$ is imposed (inital value problem), by chain rule replacing $t$ with $1-t$, CNF is can be desribed as:  

$$\frac{d\phi_t(x_1)}{dt} = v_t(t,\phi_t(x_1)), \phi_1(x_1)=x_1, x_1 \sim p(\cdot)$$  

It means that it does not matter that the simpled distribution is located at which time point.
Demo page: https://seongqjini.com/speech-enhancement-with-flow-matching-method/ -->


