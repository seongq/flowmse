# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .odesolvers import ODEsolver, ODEsolverRegistry

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'ODEsolverRegistry', 'ODEsolver', 'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_white_box_solver(
    odesolver_name,  ode, VF_fn, Y, Y_prior=None,
    T_rev=1.0, t_eps=0.03, N=30,  **kwargs
):
   
    odesolver_cls = ODEsolverRegistry.get_by_name(odesolver_name)
    
    odesolver = odesolver_cls(ode, VF_fn)

    def ode_solver(Y_prior=Y_prior):
        """The PC sampler function."""
        with torch.no_grad():
            
            if Y_prior == None:
                Y_prior = Y
            
            xt, _ = ode.prior_sampling(Y_prior.shape, Y_prior)
            if odesolver_name=="euler":
                timesteps = torch.linspace(T_rev, t_eps, N, device=Y.device)
                
            xt = xt.to(Y_prior.device)
            for i in range(len(timesteps)):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]
                    
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                
                xt = odesolver.update_fn(xt, vec_t, Y, stepsize)
            x_result = xt
            ns = len(timesteps)
            return x_result, ns
    
    return ode_solver

def get_black_box_solver(
    ode, VF_fn, y,  rtol=1e-5, atol=1e-5,  T_rev=1.0, t_eps=0.03, N=30,  method='RK45', device='cuda', **kwargs):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
     
    def ode_solver(**kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = ode.prior_sampling(y.shape, y)[0].to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = VF_fn(x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (T_rev, t_eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            return x, nfe

    return ode_solver
