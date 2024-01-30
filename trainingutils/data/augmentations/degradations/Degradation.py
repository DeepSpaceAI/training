from torch import clip as torch_clip
import torch

class Degradation():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def degrade(self, image, timestep):
        raise NotImplementedError("This degradation function hasn't been implemented yet.")
    
    def _clip(self, image, minimum=0, maximum=1):
        if type(image) != torch.Tensor:
            image = torch.Tensor(image).to(torch.float32)
        return torch_clip(image, minimum, maximum)
    
    def degrade_batch(self, batch: torch.Tensor, timestep):
        batch_size = batch.size()[0]

        degraded_images = []
        for i in range(batch_size):
            degraded_images.append(self.degrade(batch[i,:,:,:].squeeze().numpy(), timestep))
        
        return torch.stack(degraded_images, dim=0)