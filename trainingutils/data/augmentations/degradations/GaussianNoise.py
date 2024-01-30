from degradations.Degradation import Degradation
import numpy as np
from torch import tensor

class GaussianNoise(Degradation):
    def __init__(self, mean=0, std=1):
        super().__init__(mean=mean, std=std)
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        np_image = image.permute(1,2,0).numpy()
        noise = np.random.normal(self.mean, self.std, np_image.shape) + 1 / 2

        # Add noise to image
        np_image = np.sqrt((1-timestep)) * np_image + np.sqrt((timestep)) * noise
        return self._clip(tensor(np_image)).permute(2,0,1)