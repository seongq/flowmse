import abc

import torch
import numpy as np

from flowmse.util.registry import Registry


ODEsolverRegistry = Registry("ODEsolver")


class ODEsolver(abc.ABC):
    

    def __init__(self, ode, VF_fn):
        super().__init__()
        self.ode = ode        
        self.VF_fn = VF_fn
        

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@ODEsolverRegistry.register('euler')
class EulerODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t,y, stepsize, *args):
        dt = -stepsize
        vectorfield = self.VF_fn(x,t,y)
        x = x + vectorfield*dt
        
        return x
