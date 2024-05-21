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


@ODERegistry.register("otflow")
class OTFLOW(ODE):
    # Flow Matching for Generative Modelling, ICLR, Lipman et al.
    # mean_t = (1-t)x0 + tx1, sigma_t = t+sigma_min*(1-t)
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        
        return parser

    def __init__(self, sigma_min, **ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        
    def copy(self):
        return OTFLOW(self.sigma_min)

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):
        sigma_min = self.sigma_min
        return t+sigma_min*(1-t)

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
        sigma_min = self.sigma_min
        return 1-sigma_min
    


@ODERegistry.register("condflow")
class CONDFLOW(ODE):
    """ mu_t = (1-t)x0 + ty, sigma_t = sigma_min"""
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        return parser

    def __init__(self, sigma_min=0.05, **ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        
    def copy(self):
        return CONDFLOW( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return self.sigma_min*torch.ones_like(t)

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
        
        return 0

    
@ODERegistry.register("otflow_det")
class OTFLOW_DET(ODE):
    @staticmethod
    def add_argparse_args(parser):        
        
        return parser

    def __init__(self,  **ignored_kwargs):
        
        super().__init__()        

        
    def copy(self):
        return OTFLOW_DET( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return 0*torch.ones_like(t)

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
        
        return 0
    
    
    
    
    
@ODERegistry.register("flowmatching")
class FLOWMATCHING(ODE):
    #original flow matching
    #Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. International Conference on Learning Representations (ICLR), 2023.
    #mu_t = (1-t)x+ty, sigma_t = (1-t)sigma_min +t
    #t범위 0<t<=1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        
        return parser

    def __init__(self, sigma_min=0.05, **ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        
    def copy(self):
        return OTFLOW_DET( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return (1-t)*self.sigma_min + t

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
        
        return 1-self.sigma_min
    
    
@ODERegistry.register("straighCFM")
class STRAIGHTCFM(ODE):
    #Rectified flow: A marginal preserving approach to optimal transport
    #Improving and generalizing flow-based generative models with minibatch optimal transport
    #mu_t = (1-t)x+ty, sigma_t = sigma_min
    #t범위 0<t<=1인데 sigma=0이면 t=0포함
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        
        return parser

    def __init__(self, sigma_min=0.05, **ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        
    def copy(self):
        return OTFLOW_DET( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return self.sigma_min

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
        
        return 0.0
    
    
    
@ODERegistry.register("stochasticinterpolant")
class STOCHASTICINTERPOLANT(ODE):
    #Building normalizing flows with stochastic interpolants.International Conference on Learning Representations
    #mu_t = cos(1/2 pi t) x + sin(1/2 pi t) y, sigma_t = 0
    #t는 0에서 1까지 암거나 다됨 0<=t<=1
    @staticmethod
    def add_argparse_args(parser):        
        
        return parser

    def __init__(self,  **ignored_kwargs):
        
        super().__init__()        
        
        
    def copy(self):
        return OTFLOW_DET( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return torch.cos(1/2 * t * torch.pi)[:,None,None,None]*x0 + torch.sin(1/2 * t * torch.pi)[:,None,None,None]*y

    def _std(self, t):

        return 0

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
        return (-torch.sin(1/2 * t * torch.pi)[:,None,None,None]*x0 + torch.cos(1/2 * t * torch.pi)[:,None,None,None]*y)* 1/2 * torch.pi
        
    def der_std(self,t):
        
        return 0.0
    
    
    
@ODERegistry.register("SchrodingerBridge")
class SCHRODINGERBRIDGE(ODE):
    #Improving and generalizing flow-based generative models with minibatch optimal transport

    #mu_t = (1-t)x + ty,  sigma_t = sigma \sqrt{t*1(1-t)}
    #0<t<1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        return parser

    def __init__(self, sigma, **ignored_kwargs):
        
        super().__init__()        
        self.sigma = sigma
        
    def copy(self):
        return OTFLOW_DET( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return x0 * (1-t)[:,None,None,None] + y * t[:,None,None,None]

    def _std(self, t):

        return self.sigma * torch.sqrt( t *(1-t))

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y, T):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(T*torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return self.sigma* (1-2*t)/(2* torch.sqrt(t*(1-t)))[:,None,None,None]