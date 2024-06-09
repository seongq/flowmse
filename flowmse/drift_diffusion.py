# ode에 대응하는 sde를 구할 때 필요한 drift term과 diffusion term 구하기


import torch
class Drift_Diffusion():
    def __init__(ode):
        pass
    def drift(x,t,y):
        pass
    def diffusion(t):
        pass
    
    
class FLOWMATCHING_DD(Drift_Diffusion):
    def __init__(self, ode):
        super.__init__()
        assert ode.__cls__.__name__ == "FLOWMATCHING"
        self.sigma_min = ode.min()
        self.sigma_max = ode.max()
        
    def drift(x,t,y):
        return (y-x)/(1-t)
    
    def diffusion(self,t):
        return  torch.sqrt(2 * (self.max)**2 /(1-t) - 2 *self.sigma_max * (self.sigma_max - self.sigma_min))
        
        
        