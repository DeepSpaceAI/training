from degradations.Degradation import Degradation
from cv2 import GaussianBlur as opencv_blur
from torch import tensor

class GaussianBlur(Degradation):
    def __init__(self, kernel_size=(5,5), blur_min=0, blur_max=5):
        super().__init__(
            kernel_size=kernel_size,
            b_min=blur_min,
            b_max=blur_max
        )
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        blurs = int(self.b_min + (self.b_max - self.b_min) * timestep)
        blurred_image = image.permute(1,2,0).numpy()

        for i in range(blurs):
            blurred_image = opencv_blur(blurred_image, self.kernel_size, 1)

        # Convert back to torch tensor, clip values, and swap axes to be (c,h,w)
        return self._clip(tensor(blurred_image)).permute(2,0,1)