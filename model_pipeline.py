##all imports go here

import torch


## all from imports go here
from diffusers import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler      #### V2 has a different scheduler






class TextToVideoPipeline(StableDiffusionPipeline):
    def 