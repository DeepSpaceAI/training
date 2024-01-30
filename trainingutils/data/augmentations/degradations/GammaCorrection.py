from degradations.Degradation import Degradation

class GammaCorrection(Degradation):
    def __init__(self, gamma_min, gamma_max):
        super().__init__(min=gamma_min, max=gamma_max)
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        gamma_value = self.min + (self.max - self.min) * timestep
        image = image ** gamma_value
        return self._clip(image)