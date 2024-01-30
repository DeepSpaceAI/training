from degradations.Degradation import Degradation

class Exposure(Degradation):
    def __init__(self, minimum_exposure=1, maximum_exposure=1):
        super().__init__(
            minimum=minimum_exposure,
            maximum=maximum_exposure
        )
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        

        exposure_constant = timestep * (self.maximum - self.minimum) + self.minimum
        return self._clip(image * exposure_constant, 0, 1)
