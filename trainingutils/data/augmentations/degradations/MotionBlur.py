from degradations.Degradation import Degradation
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rotate
from torch import tensor

class MotionBlur(Degradation):
    def __init__(self, kernel_size=5, angle=90, blur_min=0, blur_max=5):
        super().__init__(
            kernel_size=kernel_size,
            b_min=blur_min,
            b_max=blur_max,
            angle=angle
        )
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        blurs = int(self.b_min + (self.b_max - self.b_min) * timestep)
        blurred_image = image.permute(1,2,0).numpy().copy()
            
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size-1)/2), :] = np.ones(self.kernel_size)
        kernel /= self.kernel_size

        kernel = rotate(kernel, self.angle, resize=False, mode='constant', cval=0.0)

        for i in range(blurs):
            for c in range(3):
                blurred_image[:,:,c] = convolve2d(blurred_image[:,:,c], kernel, mode='same', boundary='symm')
        
        return self._clip(tensor(blurred_image)).permute(2,0,1)