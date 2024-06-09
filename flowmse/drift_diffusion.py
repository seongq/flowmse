# ode에 대응하는 sde를 구할 때 필요한 drift term과 diffusion term 구하기


import torch
class Drift_Diffusion():
    def __init__(self,ode):
        pass
    def drift(self, x,t,y):
        pass
    def diffusion(self, t):
        pass
    
    
class FLOWMATCHING_DD(Drift_Diffusion):
    def __init__(self, ode):
        super().__init__(ode)
        assert ode.__class__.__name__ == "FLOWMATCHING"
        self.sigma_min = ode.sigma_min
        self.sigma_max = ode.sigma_max
        
    def drift(self,x,t,y):
        return (y-x)/(1-t)
    
    def diffusion(self,t):
        return  torch.sqrt(2 * (self.sigma_max)**2 /(1-t) - 2 *self.sigma_max * (self.sigma_max - self.sigma_min))
        
        
        