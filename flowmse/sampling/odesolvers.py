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


@ODEsolverRegistry.register('midpoint')
class MidpointODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t,y, stepsize, *args):
        dt = -stepsize
       
        x = x + dt*self.VF_fn(x+dt/2*self.VF_fn(x,t,y), t+dt/2, y)
        
        return x
    
@ODEsolverRegistry.register('heun')
class HeunODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t,y, stepsize, *args):
        dt = -stepsize
        current_vectorfield = self.VF_fn(x,t,y)
        x_next = x + dt * current_vectorfield
        x = x + dt/2 *(current_vectorfield+self.VF_fn(x_next,t+dt, y))
        
        return x
    
