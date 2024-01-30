from degradations.Degradation import Degradation

# This is a dummy degradation that does nothing to the image,
# but is useful for training the model when you don't want 
# the sampler to degrade the image, but still want a timestamp.
class DummyDegradation(Degradation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, image, timestep):
        return self.degrade(image,timestep)
    
    def degrade(self, image, timestep):
        return image, timestep
    
    def __str__(self):
        return "Dummy Degradation"