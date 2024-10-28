import inspect
import logging
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPFeatureExtractor,
)

from diffusers import DDPMScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import (
    AutoencoderKL,
    ImageProjection,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    KarrasDiffusionSchedulers,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging as diffusers_logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    PIL_INTERPOLATION,

)
from huggingface_hub.utils import validate_hf_hub_args

# Triton Python backend utils
try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

# Conditionally import watermark if available
if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from configs import *



class TritonPythonModel:
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    image_encoder: CLIPVisionModelWithProjection
    feature_extractor: CLIPImageProcessor
    scheduler: Union[
        DDIMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
    ]
    prompt: Union[str, List[str]]
    prompt_2: Optional[Union[str, List[str]]]
    controlnet_image: Optional[PipelineImageInput]
    cnext_model_name: str
    height: Optional[int]
    width: Optional[int]
    num_inference_steps: int
    timesteps: List[int]
    sigmas: List[float]
    denoising_end: Optional[float]
    guidance_scale: float
    negative_prompt: Optional[Union[str, List[str]]]
    negative_prompt_2: Optional[Union[str, List[str]]]
    num_images_per_prompt: Optional[int]
    eta: float
    generator: Optional[Union[torch.Generator, List[torch.Generator]]]
    latents: Optional[torch.Tensor]
    prompt_embeds: Optional[torch.Tensor]
    negative_prompt_embeds: Optional[torch.Tensor]
    pooled_prompt_embeds: Optional[torch.Tensor]
    negative_pooled_prompt_embeds: Optional[torch.Tensor]
    ip_adapter_image: Optional[PipelineImageInput]
    ip_adapter_image_embeds: Optional[List[torch.Tensor]]
    output_type: Optional[str]
    return_dict: bool
    cross_attention_kwargs: Optional[Dict[str, Any]]
    guidance_rescale: float
    original_size: Optional[Tuple[int, int]]
    crops_coords_top_left: Tuple[int, int]
    target_size: Optional[Tuple[int, int]]
    negative_original_size: Optional[Tuple[int, int]]
    negative_crops_coords_top_left: Tuple[int, int]
    negative_target_size: Optional[Tuple[int, int]]
    clip_skip: Optional[int]
    callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]]
    control_scale: float
    callback_on_step_end_tensor_inputs: List[str]
    add_watermarker: Optional[bool]
    device: str

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    @property
    def guidance_scale(self) -> Optional[float]:
        return self._guidance_scale

    @guidance_scale.setter
    def guidance_scale(self, value: Optional[float]) -> None:
        self._guidance_scale = value

    # Getter and setter for guidance_rescale
    @property
    def guidance_rescale(self) -> Optional[float]:
        return self._guidance_rescale

    @guidance_rescale.setter
    def guidance_rescale(self, value: Optional[float]) -> None:
        self._guidance_rescale = value

    # Getter and setter for clip_skip
    @property
    def clip_skip(self) -> Optional[int]:
        return self._clip_skip

    @clip_skip.setter
    def clip_skip(self, value: Optional[int]) -> None:
        self._clip_skip = value

    # Getter for do_classifier_free_guidance (no setter since it's computed)
    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1 and self.unet_configs['time_cond_proj_dim'] is None

    # Getter and setter for cross_attention_kwargs
    @property
    def cross_attention_kwargs(self) -> Optional[dict]:
        return self._cross_attention_kwargs

    @cross_attention_kwargs.setter
    def cross_attention_kwargs(self, value: Optional[dict]) -> None:
        self._cross_attention_kwargs = value

    # Getter and setter for denoising_end
    @property
    def denoising_end(self) -> Optional[float]:
        return self._denoising_end

    @denoising_end.setter
    def denoising_end(self, value: Optional[float]) -> None:
        self._denoising_end = value

    # Getter and setter for num_timesteps
    @property
    def num_timesteps(self) -> Optional[int]:
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value: Optional[int]) -> None:
        self._num_timesteps = value

    # Getter and setter for interrupt
    @property
    def interrupt(self) -> Optional[bool]:
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value: Optional[bool]) -> None:
        self._interrupt = value


    def initialize(self, args: Dict[str, str]) -> None:
        
        # current_name: str = str(Path(args["model_repository"]).parent.absolute())
        # self.scheduler_config_path = current_name + "/bls_sdxl_cnext_ip/1/scheduler/"
        # self.scheduler = DEISMultistepScheduler.from_config(self.scheduler_config_path)

        self.logger = pb_utils.Logger

        self.prompt = None
        self.prompt_2 = None
        self.negative_prompt = None
        self.negative_prompt_2 = None
        
        self.controlnet_image = None
        self.cnext_model_name = None

        self.height = None
        self.width = None
        self.num_inference_steps = 50
        self.timesteps = None
        self.sigmas = None
        self.denoising_end = None
        self.guidance_scale = 5.0

        self.num_images_per_prompt = 1
        self.eta = 0.0
        self.generator = None
        self.latents = None
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self.ip_adapter_image = None
        self.ip_adapter_image_embeds = None
        self.output_type = "pil"
        self.return_dict = True
        self.cross_attention_kwargs = None
        self.guidance_rescale = 0.0
        self.original_size = None
        self.crops_coords_top_left = (0, 0)
        self.target_size = None
        self.negative_original_size = None
        self.negative_crops_coords_top_left = (0, 0)
        self.negative_target_size = None
        self.clip_skip = None
        self.callback_on_step_end = None
        self.control_scale = 1.5
        self.dtype = torch.float32
        self.callback_on_step_end_tensor_inputs = ["latents"]
        self.add_watermarker = None

        current_name: str = str(Path(args["model_repository"]).parent.absolute())

        self.vae_configs = self.read_json(current_name + "/bls_sdxl_cnext_ip_anime/1/configs/vae_decode_config/config.json")
        self.unet_configs = self.read_json(current_name + "/bls_sdxl_cnext_ip_anime/1/configs/unet_config/config.json")
        self.text_encoder_2_configs = self.read_json(current_name + "/bls_sdxl_cnext_ip_anime/1/configs/text_encoder_2_config/config.json")
        # self.scheduler = DDPMScheduler
        self.scheduler_config_path = current_name + "/bls_sdxl_cnext_ip_anime/1/configs/scheduler/"
        self.scheduler = DDPMScheduler.from_config(self.scheduler_config_path)
        
        
        self.vae_scale_factor = 2 ** (len(self.vae_configs['block_out_channels']) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self.default_sample_size = self.unet_configs["sample_size"]
        
        add_watermarker = self.add_watermarker if self.add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        if args.get("model_instance_kind") == "GPU":
            self.device = "cuda"
        else: 
            self.device = "cpu"

        self.image_dtype = torch.float32

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image


    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )


    #Call CLIPTokenizer model.py
    def tokenizer_infer(self, prompt):
        prompt = np.array(prompt, dtype = np.object_)
        text_input = pb_utils.Tensor("TEXT", prompt)
        inference_request = pb_utils.InferenceRequest(
            model_name="tokenizer",
            requested_output_names=["input_ids"],
            inputs=[text_input],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        else:
            text_input_ids = pb_utils.get_output_tensor_by_name(
                inference_response, "input_ids"
            )
            text_input_ids: torch.Tensor = torch.from_dlpack(text_input_ids.to_dlpack())
            return text_input_ids
        
    #Call CLIPTokenizer 2 model.py
    def tokenizer_2_infer(self, prompt):
        prompt = np.array(prompt, dtype = np.object_)
        text_input = pb_utils.Tensor("TEXT", prompt)
        inference_request = pb_utils.InferenceRequest(
            model_name="tokenizer_2",
            requested_output_names=["input_ids"],
            inputs=[text_input],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        else:
            text_input_ids = pb_utils.get_output_tensor_by_name(
                inference_response, "input_ids"
            )
            text_input_ids: torch.Tensor = torch.from_dlpack(text_input_ids.to_dlpack())
            return text_input_ids
        
    #Call text encode model
    def text_encoder_infer(self, text_input_ids):
        text_input_encoder = pb_utils.Tensor("input_ids", np.array(text_input_ids))
        
        inference_request = pb_utils.InferenceRequest(
            model_name="text_encoder",
            requested_output_names=["last_hidden_state"],
            inputs=[text_input_encoder],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        
        output = pb_utils.get_output_tensor_by_name(
            inference_response, "last_hidden_state"
        )
        return output


    #Call text encode 2 model
    def text_encoder_2_infer(self, text_input_ids):
        text_input_encoder = pb_utils.Tensor("input_ids", np.array(text_input_ids))
        
        inference_request = pb_utils.InferenceRequest(
            model_name="text_encoder_2",
            requested_output_names=["last_hidden_state"],
            inputs=[text_input_encoder],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        
        output = pb_utils.get_output_tensor_by_name(
            inference_response, "last_hidden_state"
        )
        return output

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if prompt_embeds is None:
            # Prepare prompt_2
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_embeds_list = []


            #Phrase 1 - prompt
            text_input_ids = self.tokenizer_infer(prompt)
            # untruncated_ids = self.tokenizer_infer(prompt)
            prompt_embeds = self.text_encoder_infer(text_input_ids.cpu().numpy().astype(np.int32))        
            pooled_prompt_embeds = torch.from_numpy(prompt_embeds[0])
            if clip_skip is None:
                prompt_embeds = torch.from_numpy(prompt_embeds[-2])
            else:
                prompt_embeds = torch.from_numpy(prompt_embeds[-(clip_skip + 2)])
            prompt_embeds_list.append(prompt_embeds)

            #Phrase 2 - prompt
            text_input_ids = self.tokenizer_2_infer(prompt_2)
            # untruncated_ids = self.tokenizer_2_infer(prompt_2)
            prompt_embeds = self.text_encoder_2_infer(text_input_ids.cpu().numpy().astype(np.int64))        
            pooled_prompt_embeds = torch.from_numpy(prompt_embeds[0])
            if clip_skip is None:
                prompt_embeds = torch.from_numpy(prompt_embeds[-2])
            else:
                prompt_embeds = torch.from_numpy(prompt_embeds[-(clip_skip + 2)])
            prompt_embeds_list.append(prompt_embeds)
            #End prompt
            
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]
            
            #Phrase 1 - neg prompt
            negative_prompt_embeds_list = []
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer_infer(negative_prompt)
            negative_prompt_embeds = self.text_encoder_2_infer(uncond_input.input_ids.cpu().numpy().astype(np.int32))
            negative_pooled_prompt_embeds = torch.from_numpy(negative_prompt_embeds[0])
            negative_prompt_embeds = torch.from_numpy(negative_prompt_embeds[-2])
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            
            #Phrase 2 - neg prompt
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer_infer(negative_prompt)
            negative_prompt_embeds = self.text_encoder_2_infer(uncond_input.input_ids.cpu().numpy().astype(np.int64))
            negative_pooled_prompt_embeds = torch.from_numpy(negative_prompt_embeds[0])
            negative_prompt_embeds = torch.from_numpy(negative_prompt_embeds[-2])
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            #End phrase

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.dtype, device=self.device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


   
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def randn_tensor(
        self,
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
    ):
        """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # device on which tensor is created defaults to device
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cpu":
                rand_device = "cpu"
                if device != "mps":
                    self.logger.info(
                        f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                        f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                        f" slighly speed up this function by passing a generator that was created on the {device} device."
                    )
            elif gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )


    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = self.randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            self.unet_configs['addition_time_embed_dim'] * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet_configs['projection_class_embeddings_input_dim']

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


    def read_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps


    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":

        responses = []

        for request in requests:
            # client send binary data typed - convert back to string
            prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            ]
            prompt_2 = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "PROMPT_2")
                .as_numpy()
                .tolist()
            ] 
            negative_prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                .as_numpy()
                .tolist()
            ]
            negative_prompt_2 = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT_2")
                .as_numpy()
                .tolist()
            ]
            cnext_model_name = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "CNEXT_MODEL_NAME")
                .as_numpy()
                .tolist()
            ][0]   
            controlnet_image = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "CNEXT_IMAGE")
                .as_numpy()
                .tolist()
            ]
            ip_adapter_image = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "IP_ADAPTER_IMAGE")
                .as_numpy()
                .tolist()
            ]
            self.guidance_scale = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                .as_numpy()
                .tolist()
            ][0]
            self.num_inference_steps = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                .as_numpy()
                .tolist()
            ][0]
            self.control_scale = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "CNEXT_CONDITIONAL_SCALE")
                .as_numpy()
                .tolist()
            ][0]
            width = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "WIDTH")
                .as_numpy()
                .tolist()
            ]
            height = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "HEIGHT")
                .as_numpy()
                .tolist()
            ]
            seed = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                .as_numpy()
                .tolist()
            ][0] 

        #Check data input from user
        self.logger.log_info(f"############  GET REQUEST  ############")
        self.logger.log_info(f"Prompt: {prompt}")
        self.logger.log_info(f"Prompt_2: {prompt_2}")
        self.logger.log_info(f"Neg-Prompt: {negative_prompt}")
        self.logger.log_info(f"Neg-Prompt_2: {negative_prompt_2}")
        self.logger.log_info(f"Cnext model name: {cnext_model_name}")
        self.logger.log_info(f"Cnext image shape: {controlnet_image.shape}")
        self.logger.log_info(f"IP image shape: {ip_adapter_image.shape}")

        self.logger.log_info(f"guidance_scale: {self.guidance_scale.shape}")
        self.logger.log_info(f"num_inference_steps: {self.num_inference_steps.shape}")
        self.logger.log_info(f"control_scale: {self.control_scale.shape}")
        self.logger.log_info(f"width: {width.shape}")
        self.logger.log_info(f"height: {height.shape}")
        self.logger.log_info(f"seed: {seed.shape}")


        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.logger.log_info(f"############  Process  ############")
        self.logger.log_info(f"width: {width.shape}")
        self.logger.log_info(f"height: {height.shape}")

        # self._guidance_scale = self.guidance_scale
        # self._guidance_rescale = self.guidance_rescale
        # self._clip_skip = self.clip_skip
        # self._denoising_end = self.denoising_end
        # self._interrupt = False

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_images_per_prompt=self.num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=self.clip_skip,
        )

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, self.timesteps, self.sigmas
        )
        num_channels_latents = self.unet_configs['in_channels']
        
        randome_seed = np.random.RandomState(seed) if seed > 0 else np.random 
        torch_seed = randome_seed.randint(2147483647)
        torch_gen = torch.Generator().manual_seed(torch_seed)
        generator=torch_gen

        latents = self.prepare_latents(
            batch_size * self.num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2_configs['projection_dim']
        add_time_ids = self._get_add_time_ids(
            original_size,
            self.crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if self.negative_original_size is not None and self.negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                self.negative_original_size,
                self.negative_crops_coords_top_left,
                self.negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if ip_adapter_image is not None:
            #Call CLIPImageProcessor model.py
            # clip_image = self.clip_image_processor(images=ip_adapter_image, return_tensors="pt").pixel_values
            input_image_clip = Image.fromarray(ip_adapter_image)
            clip_image_processor_input = pb_utils.Tensor("IP_ADAPTER_IMAGE", input_image_clip)
            inference_request = pb_utils.InferenceRequest(
                model_name="clip_image_processor",
                requested_output_names=["pixel_values"],
                inputs=[clip_image_processor_input],
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                clip_image = pb_utils.get_output_tensor_by_name(
                    inference_response, "pixel_values"
                )
                clip_image: torch.Tensor = torch.from_dlpack(clip_image.to_dlpack())
            
            clip_image = clip_image.to(self.device, dtype=torch.float32)

            #Call Image_encoder model.onnx
            # clip_image_embeds = self.image_encoder.run(None, {'image_embedding': clip_image[0].unsqueeze(0).cpu().numpy()})
            image_encoder_input = clip_image[0].unsqueeze(0).cpu().numpy()
            image_encoder_input_triton = pb_utils.Tensor("IMAGE_EMBEDDING", image_encoder_input)
            inference_request = pb_utils.InferenceRequest(
                model_name="image_encoder",
                requested_output_names=["image_encoder"],
                inputs=[image_encoder_input_triton],
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                clip_image_embeds = pb_utils.get_output_tensor_by_name(
                    inference_response, "image_encoder"
                )
                clip_image_embeds: torch.Tensor = torch.from_dlpack(clip_image_embeds.to_dlpack())
            

            #Call Proj model.onnx
            # image_prompt_embeds = self.image_proj.run(None, {'clip_image_embeds': clip_image_embeds[0].astype(np.float32)})
            proj_image_input = clip_image_embeds[0].astype(np.float32)
            proj_iamge_input_triton = pb_utils.Tensor("CLIP_IMAGE_EMBEDS", proj_image_input)
            inference_request = pb_utils.InferenceRequest(
                model_name="proj",
                requested_output_names=["image_prompt_embeds"],
                inputs=[proj_iamge_input_triton],
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                image_prompt_embeds = pb_utils.get_output_tensor_by_name(
                    inference_response, "image_prompt_embeds"
                )
                image_prompt_embeds: torch.Tensor = torch.from_dlpack(image_prompt_embeds.to_dlpack())
            
            #Call Proj model.onnx
            # uncond_image_prompt_embeds = self.image_proj.run(None, {'clip_image_embeds': torch.zeros_like(torch.tensor(clip_image_embeds[0])).cpu().numpy().astype(np.float32)})
            proj_uncond_image_input = clip_image_embeds[0].astype(np.float32)
            proj_uncond_iamge_input_triton = pb_utils.Tensor("CLIP_IMAGE_EMBEDS", proj_uncond_image_input)
            inference_request = pb_utils.InferenceRequest(
                model_name="proj",
                requested_output_names=["image_prompt_embeds"],
                inputs=[proj_uncond_iamge_input_triton],
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                uncond_image_prompt_embeds = pb_utils.get_output_tensor_by_name(
                    inference_response, "image_prompt_embeds"
                )
                uncond_image_prompt_embeds: torch.Tensor = torch.from_dlpack(uncond_image_prompt_embeds.to_dlpack())
            

            image_prompt_embeds = torch.from_numpy(image_prompt_embeds[0]).to(self.device)
            uncond_image_prompt_embeds = torch.from_numpy(uncond_image_prompt_embeds[0]).to(self.device)

            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([torch.zeros_like(negative_prompt_embeds), uncond_image_prompt_embeds], dim=1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size * self.num_images_per_prompt, 1)

        if controlnet_image is not None and self.controlnet is not None:
            prepare_image_input = Image.fromarray(controlnet_image)
            controlnet_image = self.prepare_image(
                prepare_image_input,
                width,
                height,
                batch_size,
                self.num_images_per_prompt,
                self.device,
                self.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
       
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]



        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            print('Step:', i)
            if self.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            #Call cnext model.onnx
            # controls = self.controlnet.run(None, {'controlnext_image': controlnet_image.cpu().numpy(),
            #             'timestep': t.unsqueeze(0).cpu().numpy().astype(np.float32),})

            input_cnext = [
                pb_utils.Tensor("controlnext_image", controlnet_image.cpu().numpy().astype(np.float32)),
                pb_utils.Tensor("timestep",  t.unsqueeze(0).cpu().numpy().astype(np.float32)),
            ]

            inference_request = pb_utils.InferenceRequest(
                model_name=cnext_model_name, 
                requested_output_names=['sample'],  
                inputs=input_cnext,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message())
            else:
                controls = pb_utils.get_output_tensor_by_name(
                    inference_response, "sample"
                )
                controls: torch.Tensor = torch.from_dlpack(controls.to_dlpack())
            
            scale = torch.tensor([self.control_scale])


            #Call the unet model.onnx
            input_unet = [
                pb_utils.Tensor("control_out", latent_model_input.cpu().numpy().astype(np.float32)),
                pb_utils.Tensor("timestep",  t.unsqueeze(0).cpu().numpy().astype(np.float32)),
                pb_utils.Tensor("encoder_hidden_state",  prompt_embeds.cpu().numpy().astype(np.float32)),
                pb_utils.Tensor("control_out",  controls[0].astype(np.float32)),
                pb_utils.Tensor("control_scale",  scale.cpu().numpy().astype(np.float32))
            ]

            inference_request = pb_utils.InferenceRequest(
                model_name=cnext_model_name, 
                requested_output_names=['predict_noise'],  
                inputs=input_unet,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message())
            else:
                noise_pred = pb_utils.get_output_tensor_by_name(
                    inference_response, "predict_noise"
                )
                noise_pred: torch.Tensor = torch.from_dlpack(noise_pred.to_dlpack())
            
            noise_pred = torch.from_numpy(noise_pred[0]).to(self.device)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)


            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if self.callback_on_step_end is not None:
                callback_kwargs = {}
                for k in self.callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = self.callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )
                add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

        if not self.output_type == "latent":
            '''needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)'''

            has_latents_mean = hasattr(self.vae_configs, "latents_mean") and self.vae_configs['latents_mean'] is not None
            has_latents_std = hasattr(self.vae_configs, "latents_std") and self.vae_configs['latents_std'] is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae_configs['latents_mean']).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae_configs['latents_std']).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae_configs['scaling_factor'] + latents_mean
            else:
                latents = latents / self.vae_configs['scaling_factor']
            #Call VAE model.onnx
            # image = self.vae.run(None, {'latent_sample': latents.cpu().numpy()})[0]
                input_vae = [
                    pb_utils.Tensor.from_dlpack(
                        "latent_sample", latents.cpu().numpy().astype(np.float32)
                    )
                ]
                self.logger.log_warn(f"latent_sample for vae: {latents.shape}")
                self.logger.log_warn(f"latent_sample for vae: {type(latents)}")

                inference_request = pb_utils.InferenceRequest(
                    model_name="vae_decode",
                    requested_output_names=["sample"],
                    inputs=input_vae,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(inference_response, "sample")
                    image: torch.Tensor = torch.from_dlpack(output.to_dlpack())
        else:
            image = latents
        
        image = image.cpu().numpy().astypee(np.float32)

        tensor_output = [pb_utils.Tensor("IMAGES", image)]
        responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
    
    def finalize(self) -> None:
        """
        Called when the model is being unloaded from memory.
        """
        pass











