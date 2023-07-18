##all imports go here

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch


## all from imports go here
from diffusers import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler      #### V2 has a different scheduler


from diffusers.models import AutoencoderKL, UNet2DConditionModel ##

from transformers import CLIPImageProcessor
from transformers import  CLIPTextModel
from transformers import CLIPTokenizer




class TextToVideoPipeline(StableDiffusionPipeline):
    def __init__(self, vae: AutoencoderKL, 
                 text_encoder: CLIPTextModel, 
                 tokenizer: CLIPTokenizer, 
                 unet: UNet2DConditionModel, 
                 scheduler: KarrasDiffusionSchedulers, 
                 safety_checker: StableDiffusionSafetyChecker, 
                 feature_extractor: CLIPImageProcessor, 
                 requires_safety_checker: bool = True):
        
        #inheriting 
        super().__init__(vae, 
                         text_encoder, 
                         tokenizer, 
                         unet, 
                         scheduler, 
                         safety_checker, 
                         feature_extractor, 
                         requires_safety_checker)
    


    ## Exposing Encoding
    def expose_encode_prompt(
        self,
        prompt,                                                     #prompt to be encoded
        device,                                                     #torch device
        num_images_per_prompt,                                      #no of images per prompt(int)
        do_classifier_free_guidance,                                #
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        return
    

    def expose_latents():
        return




    #custom tokenizer
    def custom_tokenizer(self, prompt):
        custom_cache={}
        tokenizer=CLIPTokenizer

        

    
        
        
        