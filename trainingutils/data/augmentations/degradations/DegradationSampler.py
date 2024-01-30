import numpy as np

class DegradationSampler():
    def __init__(self, max_timesteps):
        self.max_timesteps = max_timesteps
        self.alphas = np.linspace(0,1,max_timesteps)
        self.degradations = {}
    
    def __call__(self, image, timestep=None):
        return self.generate_sample(image, timestep)

    def generate_sample(self, image, timestep=None, batch=False):
        if timestep is None:
            timestep = np.random.randint(0, self.max_timesteps)
        
        assert timestep < self.max_timesteps

        timestep_alpha = self.alphas[timestep]

        degraded_image = image.clone()
        for degradation in self.degradations.values():
            if batch:
                degraded_image = degradation.degrade_batch(degraded_image, timestep_alpha)
            else:
                degraded_image = degradation(degraded_image, timestep_alpha)
        
        return degraded_image, timestep_alpha, timestep

    def generate_samples(self, image, num_samples):
        samples = []
        timesteps = []

        samples = [int(x) for x in np.linspace(0, self.max_timesteps, num_samples)]

        for sample_timestep in samples:
            sample, _, timestep = self.generate_sample(image, sample_timestep)
            samples.append(sample)
            timesteps.append(timestep)
        
        return samples, timesteps

    def __str__(self):
        return f"Degradations: {','.join([str(degradation) for degradation in self.degradations])}"
    
    def add(self, degradation):
        self.degradations[str(degradation)] = degradation
    
    def add_multiple(self, degradations):
        self.degradations.extend(degradations)

    def remove(self, degredation):
        removed_degradation = False

        for degrad_index in range(len(self.degradations)):
            if str(self.degradations[degrad_index]) == degredation:
                self.degradations.pop(degrad_index)
                removed_degradation = True
                break
        
        return removed_degradation
