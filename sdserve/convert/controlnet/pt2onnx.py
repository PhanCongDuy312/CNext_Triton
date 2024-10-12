import torch
from pathlib import Path
import inspect

from diffusers import ControlNetModel

from sdserve.models.cnet.flex import FlexControlNet
from sdserve.convert.onnx import onnx_export, convert_to_fp16

class CnetConverter:
    def __init__(self,
        model_path: str, output_path: str, opset: int, fp16: bool, attention_slicing: str | int | None    
    ) -> None:
        # load from any huggingface model 
        self.controlnet = ControlNetModel.from_pretrained(model_path)
        # check if the model is SDXL or SD15
        self.is_sdxl = True
        if len(self.controlnet.config.down_block_types) == 4:
            self.is_sdxl = False
        # slice the attention heads
        if attention_slicing is not None:
            print("Attention slicing: " + attention_slicing)
            self.controlnet.set_attention_slice(attention_slicing)
        # quantize the model
        self.dtype=torch.float32 if not fp16 else self.dtype
        self.device = "cpu"
        # set output path
        output_path = Path(output_path)
        self.cnet_path = output_path / "controlnet" / "model.onnx"
        # setstuff
        self.opset = opset
        self.fp16 = fp16

    @torch.no_grad()
    def convert(self, scheduler):
        """
        Convert the controlnet model to ONNX format

        Args:
            model_path (str): path to the model
            output_path (str): path to save the ONNX model
            opset (int): ONNX opset version
            fp16 (bool): convert the model to FP16
            attention_slicing (str): When "auto", input to the attention heads is halved, so attention is computed in two steps. If "max", maximum amount of memory is saved by running only one slice at a time. If a number is provided, uses as many slices as attention_head_dim // slice_size. In this case, attention_head_dim must be a multiple of slice_size. When this option is enabled, the attention module splits the input tensor in slices to compute attention in several steps. This is useful for saving some memory in exchange for a small decrease in speed.
        """
        # warp model
        dump_controlnet = FlexControlNet(self.controlnet, scheduler)
        
        # get the signature of the forward method
        inputs_args = inspect.getfullargspec(dump_controlnet.forward).args
        inputs_args.remove("self")  # remove self

        # get positional arguments

        if not self.is_sdxl:
            dump_inputs = (
                torch.randn(2, 4, 64, 64).to(device=self.device, dtype=self.dtype),
                torch.randn(1).to(device=self.device, dtype=self.dtype),
                torch.randn(2, 77, 768).to(device=self.device, dtype=self.dtype),
                torch.randn(2, 3, 512,512).to(device=self.device, dtype=self.dtype),
                torch.randn(1).to(device=self.device, dtype=self.dtype), # controlnet_cond
                None, None, None, # add_text_embeds, add_time_ids, controlnet_keep_i
                False, False,False # guess_mode, do_classifier_free_guidance, is_sdxl
            )
            outputs_names = [
                "down_block_0",
                "down_block_1",
                "down_block_2",
                "down_block_3",
                "down_block_4",
                "down_block_5",
                "down_block_6",
                "down_block_7",
                "down_block_8",
                "mid_block_res_sample",
            ]
            extra_axes = {
                "down_block_0": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_1": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_2": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_3": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_4": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_5": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_6": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_7": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_8": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "mid_block_res_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            }
        else:
            dump_inputs = (
                torch.randn(2, 4, 128, 128, dtype=self.dtype, device=self.device),
                torch.randn(1).to(device=self.device, dtype=self.dtype), # timestep
                torch.randn(2, 77, 2048, dtype=self.dtype, device=self.device), # encoder_hidden_states
                torch.randn(2, 3, 1024, 1024, dtype=self.dtype, device=self.device), # cond_image
                torch.randn(1).to(device=self.device, dtype=self.dtype), # cond_scale
                torch.randn(2, 1280, dtype=self.dtype, device=self.device), # add_text_embeds
                torch.randn(2, 6, dtype=self.dtype, device=self.device), # add_time_ids
                None, # controlnet_keep_i
                False, # guess_mode
                False, # do_classifier_free_guidance
                True, # is_sdxl
            )
            outputs_names = [
                "down_block_0",
                "down_block_1",
                "down_block_2",
                "down_block_3",
                "down_block_4",
                "down_block_5",
                "down_block_6",
                "down_block_7",
                "down_block_8",
                "down_block_9",
                "down_block_10",
                "down_block_11",
                "mid_block_res_sample",
            ]
            extra_axes = {
                "down_block_0": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_1": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_2": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_3": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_4": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_5": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_6": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_7": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_8": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_9": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_10": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "down_block_11": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "mid_block_res_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            }


        print("Exporting controlnet to ONNX...")
        onnx_export(
            dump_controlnet,
            model_args=dump_inputs,
            output_path=self.cnet_path,
            ordered_input_names=inputs_args,
            output_names=outputs_names,  #* has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
                "controlnet_cond": {0: "batch", 2: "height", 3: "width"},
            }.update(extra_axes), #* timestep and condition_scale are not dynamic axes
            opset=self.opset,
        )
        if self.fp16:
            print("Converting controlnet to FP16...")
            cnet_path_model_path = str(self.cnet_path.absolute().as_posix())
            convert_to_fp16(cnet_path_model_path)
        print("Controlnet exported to ONNX successfully at: " + str(self.cnet_path.absolute().as_posix()))