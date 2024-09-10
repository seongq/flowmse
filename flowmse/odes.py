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
    
    
    
    



######################여기 밑에 것이 학습할 대상임##############


@ODERegistry.register("flowmatching")
class FLOWMATCHING(ODE):
    #original flow matching
    #Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. International Conference on Learning Representations (ICLR), 2023.
    #mu_t = (1-t)x+ty, sigma_t = (1-t)sigma_min +t
    #t범위 0<t<=1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max",type=float, default=1.0 , help="The maximum sigma to use. 1 by default") 
        parser.add_argument("--T_sampling",type=float, default=1.0)
        return parser

    def __init__(self, sigma_min=0.05, sigma_max = 1.0, T_sampling=1.0,**ignored_kwargs):
        
        super().__init__()        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T_sampling = T_sampling
        
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
        return STRAIGHTCFM(self.sigma_min)

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return self.sigma_min + torch.zeros_like(t)

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
        return STOCHASTICINTERPOLANT( )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return torch.cos(1/2 * t * torch.pi)[:,None,None,None]*x0 + torch.sin(1/2 * t * torch.pi)[:,None,None,None]*y

    def _std(self, t):

        return 0 + torch.zeros_like(t)

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
    
    
    
@ODERegistry.register("schrodingerBridge")
class SCHRODINGERBRIDGE(ODE):
    #Improving and generalizing flow-based generative models with minibatch optimal transport

    #mu_t = (1-t)x + ty,  sigma_t = sigma \sqrt{t*1(1-t)}
    #0<t<1
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma", type=float, default=0.973862, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--T", type=float, default=0.999, help="Reverse starting point of Schrodinger bridge")
        return parser

    def __init__(self, sigma, T=0.999,**ignored_kwargs):
        
        super().__init__()        
        self.sigma = sigma
        self.T_rev = T
        
    def copy(self):
        return SCHRODINGERBRIDGE(self.sigma, self.T_rev )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return x0 * (1-t)[:,None,None,None] + y * t[:,None,None,None]

    def _std(self, t):

        return self.sigma * torch.sqrt( t *(1-t))

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T_rev*torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return (self.sigma* (1-2*t)/(2* torch.sqrt(t*(1-t))))[:,None,None,None]
    
    
    
@ODERegistry.register("flowmatchingconvex")
class FLOWMATCHINGCONVEX(ODE):
    
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma", type=float,  default=0.487, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--n", type=int,  default=2, help="t**(n/2), n")
        parser.add_argument("--T", type=float, default=0.999)
        
        return parser

    def __init__(self, sigma=0.487, n=2, T=0.999, **ignored_kwargs):
        
        super().__init__()        
        self.sigma = sigma
        self.n = n
        self.T_rev = T
        
        
    def copy(self):
        return FLOWMATCHINGCONVEX(self.sigma, self.n, self.T_rev )

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return (torch.sqrt(t)**self.n)*self.sigma

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T_rev*torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return self.n * 1/2 * ((t[:,None,None,None])**(self.n/2-1)) * self.sigma
    
    
    
@ODERegistry.register("flowmatchingconcave")
class FLOWMATCHINGCONCAVE(ODE):
    
    @staticmethod
    def add_argparse_args(parser):        
        parser.add_argument("--sigma", type=float,  default=0.487, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--T", type=float, default=0.999)
        parser.add_argument("--n", type=int, default=2, help="(t*(2-t))**(n/2), n")
        
        return parser

    def __init__(self, sigma=0.487,n=2, T=0.999, **ignored_kwargs):
        
        super().__init__()        
        self.sigma = sigma
        self.n = n
        self.T_rev = T
        
        
    def copy(self):
        return FLOWMATCHINGCONCAVE(self.sigma, self.n, self.T_rev)

    def ode(self,x,t,*args):
        pass    
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):

        return self.sigma*(t*(2-t)**(self.n/2))

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T_rev* torch.ones((y.shape[0],), device=y.device)) #inference시 사이즈 맞추기 위함
        z = torch.randn_like(y)
        
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        return self.sigma * self.n * (1-t) * ( (t* (2-t))**(self.n/2-1))[:,None,None,None]
    
    

@ODERegistry.register("bbed")
class BBED(ODE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--k", type=float, default = 2.6, help="base factor for diffusion term") 
        parser.add_argument("--theta", type=float, default = 0.52, help="root scale factor for diffusion term.")
        parser.add_argument("--T", type=float, default=0.999)
        return parser

    def __init__(self,  k, theta,T , **kwargs):
        """Construct an Brownian Bridge with Exploding Diffusion Coefficient SDE with parameterization as in the paper.
        dx = (y-x)/(Tc-t) dt + sqrt(theta)*k^t dw
        """
        super().__init__()
        self.k = k
        self.logk = np.log(self.k)
        self.theta = theta
        self.Eilog = sc.expi(-2*self.logk)
        self.T_rev = T


    def copy(self):
        return BBED(self.k, self.theta, self.T_rev)

    def ode(self, x, t, *args):
        pass


    def _mean(self, x0, t, y):
        time = (t)[:, None, None, None]
        mean = x0*(1-time) + y*time
        return mean

    def _std(self, t):
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2*(t_np-1)*self.logk) - self.Eilog
        h = 2*self.k**2*self.logk
        var = (self.k**(2*t_np)-1+t_np) + h*(1-t_np)*Eis
        var = torch.tensor(var).to(device=t.device)*(1-t)*self.theta
        return torch.sqrt(var)

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T_rev*torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z
    
    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        
        # t_np = t.cpu().detach().numpy()
        # Eis = sc.expi(2*(t_np-1)*self.logk) - self.Eilog
        # h = 2*self.k**2*self.logk
        # var = (self.k**(2*t_np)-1+t_np) + h*(1-t_np)*Eis
        # var = torch.tensor(var).to(device=t.device)*(1-t)*self.theta
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2*(t_np-1)*self.logk) - self.Eilog
        Eis = torch.tensor(Eis).to(t.device)
        h = 2*self.k**2*self.logk
        pre_var = (self.k**(2*t)-1+t) + h*(1-t)*Eis
        var = pre_var*(1-t)*self.theta
        std = torch.sqrt(var).to(t.device)
        
        dEis = self.k**(2*(t-1))/(t-1)
        dpre_var = 2*self.logk*(self.k**(2*t))+1 -h*Eis + h*(1-t)*dEis
        dvar = dpre_var*(1-t)*self.theta-pre_var*self.theta
        dstd = 1/(2*std) * dvar
        
        return dstd[:,None,None,None]



@ODERegistry.register("ouve")
class OUVESDE(ODE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--theta", type=float, default=1.5, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max,  **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__()
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max)

    def ode(self, x, t, y):
        pass

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        #z = torch.zeros_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def der_mean(self,x0,t,y):
        theta = self.theta
        return theta * torch.exp(-theta *t)[:,None,None,None]*(y-x0)        
    def der_std(self,t):
        std = self._std(t)
        logsig = self.logsig
        sigma_min = self.sigma_min
        theta = self.theta
                
        return (1/std * (sigma_min**2)/(theta+logsig) *(logsig* torch.exp(2*logsig*t) + theta * torch.exp(-2 * theta *t)) * logsig) [:,None,None,None]
    