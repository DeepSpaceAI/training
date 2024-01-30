from degradations.Degradation import Degradation
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_HSV2RGB
import cv2
from torch import tensor

class HueShift(Degradation):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image, timestep=0):
        return self.degrade(image, timestep)
    
    def degrade(self, image, timestep):
        np_image = image.permute(1,2,0).numpy()
        hsv_image = cvtColor(np_image, COLOR_RGB2HSV)

        # Add noise to image
        hsv_image[:,:,0] = (hsv_image[:,:,0] + timestep * 340) % 360
        np_image = cvtColor(hsv_image, COLOR_HSV2RGB)

        return self._clip(tensor(np_image)).permute(2,0,1)