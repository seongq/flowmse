# SGMSE - Brownian Bridge with Exponential Diffusion Coefficient

This repository contains the official PyTorch implementations for the 2023 paper:

- *Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement* [1]

This repository builds upon our previous work, that can be found here https://github.com/sp-uhh/sgmse


## Installation
- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.



## Training
Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [1]) can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files. To reproduce results in [1] you could use the following training settings on the wsj0-chime3 dataset:

```bash
python train.py --base_dir <your_base_dir> --batch_size 16 --backbone ncsnpp --sde bbed --t_eps 0.03 --gpus 1 --num_eval_files 10 --spec_abs_exponent 0.5 --spec_factor 0.15 --loss_abs_exponent 1 --loss_type mse --k 2.6 --theta 0.51
```
To get the training set, we refer to [https://github.com/sp-uhh/sgmse](https://github.com/sp-uhh/sgmse/tree/main/preprocessing) and execute create_wsj0_chime3.py.






To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.

## Evaluation

To evaluate on a test set, run
```bash
python eval.py --test_dir <your_test_dir> --folder_destination <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
```

to generate the enhanced .wav files. For instance,
```bash
python eval.py --test_dir <your_test_dir> --folder_destination <your_enhanced_dir> --ckpt <path_to_model_checkpoint> --N 30 --reverse_starting_point 0.5 --force_N 15
```
starts enhancement from 0.5 with 15 reverse steps. This would be the result of Tab. 1 last row in [1], when the provided checkpoint (download from here https://drive.google.com/file/d/1_h7pH6o-j7GV_E69SbRQF2BMRlC8tmz_/view?usp=share_link) is loaded in the checkpoint folder. This is the checkpoint that was used to produce the results in [1].




## Citations / References

We kindly ask you to cite our paper (can be found on https://arxiv.org/abs/2302.14748) in your publication when using any of our research or code:

>[1] Bunlong Lay, Simon Welker, Julius Richter and Timo Gerkmann. *Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement*, ISCA Interspeech, 2023.
# flowmse
# flowmse
# flowmse
# flowmse
# flowmse
