from degradations.Degradation import Degradation

class SpeckleNoise(Degradation):
    def __init__(self, mean=0, std=1):
        super().__init__(mean=mean, std=std)
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        print("Implement Degradation")
        pass