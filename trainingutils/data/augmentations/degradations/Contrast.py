from degradations.Degradation import Degradation

class Contrast(Degradation):
    def __init__(self, minimum_contrast, maximum_contrast):
        super().__init__(
            maximum_contrast=maximum_contrast,
            minimum_contrast=minimum_contrast
        )
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        contrast_adjustment = self.minimum_contrast + (self.maximum_contrast - self.minimum_contrast) * timestep
        image = (image - 0.5) * contrast_adjustment + 0.5
        return self._clip(image)