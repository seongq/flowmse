"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings
import math
import scipy.special as sc
import numpy as np
from flowmse.util.tensors import batch_broadcast
import torch

from flowmse.util.registry import Registry


ODERegistry = Registry("ODE")
class ODE(abc.ABC):
    """ODE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):        
        super().__init__()
        

    
    @abc.abstractmethod
    def ode(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass


    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass


    @abc.abstractmethod
    def copy(self):
        pass



######################여기 밑에 것이 학습할 대상임##############


@ODERegistry.register("flowmatching")
class FLOWMATCHING(ODE):
    #original flow matching
    #Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. International Conference on Learning Representations (ICLR), 2023.
    #mu_t = (1-t)x+ty, sigma_t = (1-t)sigma_min +t
    #t범위 0<t<=1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma_min", type=float, default=0.00, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma_max",type=float, default=0.5 , help="The maximum sigma to use. 1 by default") 
        return parser

    def __init__(self, sigma_min=0.00, sigma_max =0.5, **ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        
    def copy(self):
        return FLOWMATCHING(self.sigma_min,self.sigma_max  )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return (1-t)*self.sigma_min + t*self.sigma_max

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return self.sigma_max-self.sigma_min
    
    
    