"""MIT License

Copyright (c) [2025] [TBG Tobias Laarmann]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

All Code under Copyright (c) [2025] [TBG Tobias Laarmann] exept:

Part of the code includes ModelSampleFlux Normalized from 42lux:

MIT License

Copyright (c) [2024] [42lux]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MIT License

Part of the code includes parts of lyingsigma from Jonseed:

Copyright (c) 2024 Jonseed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import comfy.utils
import latent_preview
import torch.nn.functional as F
import comfy
import math
from comfy.samplers import KSamplerX0Inpaint
from comfy.samplers import KSAMPLER
import comfy.utils
from comfy import model_management
from server import PromptServer
import base64
from PIL import Image
import io
import numpy as np
import json
from typing import Dict, Any, Tuple
from comfy.samplers import cast_to_load_options
from comfy_extras.nodes_custom_sampler import KSamplerSelect
ksampler_instance = KSamplerSelect()
if hasattr(KSamplerSelect, "execute"):
    KSamplerSelect_execute = ksampler_instance.execute
elif hasattr(KSamplerSelect, "get_sampler"):
    KSamplerSelect_execute = ksampler_instance.get_sampler



class TBG_FluxKontextStabilizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "stabilize"
    CATEGORY = "TBG/Takeaways"

    def stabilize(self, sigmas):
        # Consistent Position Sigma
        sigma_b = torch.tensor([
            1.0000, 0.9910, 0.9753, 0.9547, 0.9295, 0.8994,
            0.8643, 0.8236, 0.7770, 0.7238, 0.6636, 0.5965,
            0.5223, 0.4419, 0.3571, 0.2711, 0.1877, 0.1130,
            0.0527, 0.0138, 0.0000
        ])

        sigma_a = torch.tensor([1.0000, 0.9836, 0.9660, 0.9471, 0.9266, 0.9045, 0.8805, 0.8543, 0.8257,
                0.7942, 0.7595, 0.7210, 0.6780, 0.6297, 0.5751, 0.5128, 0.4412, 0.3579,
                0.2598, 0.1425, 0.0000])

        # Get first 6 steps from sigma_a
        head = sigma_a[:6]
        # Threshold is the 6th value (index 5)
        threshold = sigma_a[5].item()
        # Filter sigmas input to keep only values less than threshold
        filtered_b = sigmas[sigmas < threshold]
        # Combine head and filtered_b
        result = torch.cat([head, filtered_b], dim=0)

        return (result, )


class ModelSamplingFluxGradual:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL", {"tooltip": "The model to apply sampling adjustments to"}),
                     "max_shift": ("FLOAT", {"default": 0.87,"min": -100.0,"max": 100.0,"step": 0.001,"tooltip": "Maximum shift Img2Img lower value retain better the materials 0.5-0.9"}),
                     "base_shift": ("FLOAT", {"default": 0.5,"min": -100.0,"max": 100.0,"step": 0.001,"tooltip": "Base shift  Img2Img lower value retain better the materials 0.1"}),
                     "latent": ("LATENT", { "tooltip": "The latent to calculate image dimensions from"}),
                     "GradualModelShift": ("FLOAT", {"default": 0,"min": -1,"max": 2,"step": 0.001,"tooltip": "Like Creativity in Magnific. Interpolates between a ImageSize ModelSampelFluxNormalized 0 to ModelSampelFlux 1. 0"}),
                      }
               }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "TBG/Takeaways"

    FUNCTION = "patch"

    def patch(self, model, latent, max_shift: float, base_shift: float, GradualModelShift) -> tuple:
        lc = latent.copy()
        size = lc["samples"].shape[3], lc["samples"].shape[2]
        size = size[0] * 8, size[1] * 8
        width, height = size

        MP =  (width * height) / 1000000
        GradualShift = GradualModelShift #+ (0.02 * MP - 0.084)

        m = model.clone()
        adjusted_max_shift = (base_shift - max_shift) / (256 - ((width * height) / 256)) * 3840 + base_shift
        interpolated_max_shift = GradualShift * max_shift + (1 - GradualShift) * adjusted_max_shift
        x1 = 256
        x2 = 4096
        mm = (interpolated_max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)

        #take Gradualshift and make an interpolation of ModelSamplingFlux and m

        return (m,)


#------------------------------------------------------------------------------------------

class  PolyExponentialSigmaAdder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", {"tooltip": "The model to apply sampling adjustments to"}),
                     "polyexponential_multipier": ("FLOAT", {"default": 0.0, "min": -5, "max": 5, "step": 0.001,"tooltip": "finetuninng sigmas"}),
                     "curves_rigidity": ("FLOAT", {"default": 1.5, "min": -10, "max": 10, "step": 0.001,"tooltip": "finetuninng sigmas"}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "TBG/Takeaways"

    FUNCTION = "get_sigmas"


    def get_sigmas(self, sigmas, curves_rigidity,polyexponential_multipier):
        rho = curves_rigidity #3
        polyexponential_multipier = -polyexponential_multipier
        ramp = torch.linspace(1, 0, len(sigmas), device='cpu') ** rho
        sigma_max = sigmas[0]
        sigma_min = 0.0001
        if polyexponential_multipier:
            polyexponential_multipier = 1.001-(-polyexponential_multipier)
            sigmas_polyexponential = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
            flipped_tensor = torch.flip(sigmas_polyexponential, [0])
            sigmas_polyexponential = sigma_max + sigma_min - flipped_tensor
            sigmas = polyexponential_multipier * sigmas + (1 - polyexponential_multipier) * sigmas_polyexponential

        return (sigmas,)

#------------------------------------------------------------------------------------------------


class BasicSchedulerNormalized:

    DENOISE_METHODS = [
        'default',
        'default short ',
        'normalized',
        'normalized advanced',
        'multiplyed',
        'multiplyed normalized'
    ]

    @classmethod
    def INPUT_TYPES(self):
        return {"required":
                    {"model": ("MODEL",),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "denoise_method": (self.DENOISE_METHODS, { "label": "DENOISE_METHODS", "default": 'normalized'}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "TBG/Takeaways"

    FUNCTION = "get_sigmas"



    def get_sigmas(self, model, scheduler, steps, denoise, denoise_method):
        total_steps=steps
        if denoise == 1:
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
            return (sigmas, )
        if denoise_method == "default":
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                aumented_steps = int(steps/denoise)
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, aumented_steps).cpu()
            # Cuts out the lowSteps from the totalSteps
            sigmas = sigmas[-(total_steps):]

        if denoise_method == "multiplyed":
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                sigmas = sigmas * denoise

        if denoise_method == 'multiplyed normalized':
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                max_sigma = sigmas.max()
                scale_factor = denoise / max_sigma
                sigmas = sigmas * scale_factor

        if denoise_method == "normalized":
            # first get default sigmas
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                aumented_steps = int(steps/denoise)
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, aumented_steps).cpu()
            # Cuts out the lowSteps from the totalSteps
            sigmas = sigmas[-(total_steps):]

            # second scale the sigmas to max value = denoise value
            max_sigma = sigmas.max()
            scale_factor = denoise / max_sigma
            sigmas = sigmas * scale_factor

        if denoise_method == "default short ":
            # first get default sigmas
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                reduced_steps = math.ceil(steps*denoise)
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
            # Cuts out the lowSteps from the totalSteps
            sigmas = sigmas[-(reduced_steps):]

        if denoise_method == "normalized advanced":
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
             # Fund the step where denoise = restnoise
                differences = torch.abs(sigmas - denoise)
                closest_step = torch.argmin(differences).item()
             # Cut the tail of the sigmas starting from closest_step
                sliced_sigmas = sigmas[closest_step:]

                if sliced_sigmas.shape[0] < 2:
                    # Interpolation needs at least 2 points
                    return (torch.FloatTensor([]),)

                # Reshape for 1D linear interpolation
                sliced_sigmas = sliced_sigmas.view(1, 1, -1)

                # Interpolate to total_steps
                interpolated_sigmas = F.interpolate(
                    sliced_sigmas, size=total_steps, mode='linear', align_corners=True
                ).view(-1)
                # Scale to restnoise = denoise
                max_sigma = interpolated_sigmas.max()
                scale_factor = denoise / max_sigma
                sigmas = interpolated_sigmas * scale_factor
                # Normalize so that max becomes `denoise`
                max_sigma = interpolated_sigmas.max()
                if max_sigma > 0:
                    scale_factor = denoise / max_sigma
                    sigmas = interpolated_sigmas * scale_factor
                else:
                    return (torch.FloatTensor([]),)

        return (sigmas, )





#LyingSigmaSampler
def Log_Sigma_Sampler(
    model,
    x,
    sigmas,
    *,
    lss_wrapped_sampler,
    lss_dishonesty_factor,
    lss_lerpfactor,
    lss_startend_percent,
    **kwargs,
):

    start_percent, end_percent = lss_startend_percent
    ms = model.inner_model.inner_model.model_sampling
    start_sigma, end_sigma = (
        round(ms.percent_to_sigma(start_percent), 4),
        round(ms.percent_to_sigma(end_percent), 4),
    )
    del ms

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        adjusted_sigma = sigma
        if end_sigma <= sigma_float <= start_sigma:
            # Find index of nearest element
            index = torch.argmin(torch.abs(sigmas.to(sigma.device) - sigma))
            print(f"index {index}")
            LogSigmas = lss_dishonesty_factor[index]
            adjusted_sigma = torch.lerp(torch.tensor([LogSigmas], device=sigma.device),sigma,lss_lerpfactor)
            print(f"lss_dishonesty_factor[index] {lss_dishonesty_factor[index]}, sigma {sigma}, siggma_float {sigma_float} start_sigma {start_sigma}")
        return model(x, adjusted_sigma, **extra_args)


    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return lss_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **lss_wrapped_sampler.extra_options,
    )

class LogSigmaSamplerNode:
    CATEGORY = "TBG/Takeaways"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "LogSigmas" : ("SIGMAS",),
                "lerpfactor": (
                    "FLOAT",
                    {
                        "default": -0.05,
                        "min": -0.999,
                        "step": 0.01,
                        "tooltip": "Multiplier for sigmas passed to the model. -0.05 means we reduce the sigma by 5%.",
                    },
                ),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    @classmethod
    def go(cls, sampler, LogSigmas,     lerpfactor, *, start_percent=0.0, end_percent=1.0):
        return (
            KSAMPLER(
                Log_Sigma_Sampler,
                extra_options={
                    "lss_wrapped_sampler": sampler,
                    "lss_dishonesty_factor": LogSigmas,
                    "lss_lerpfactor" : lerpfactor,
                    "lss_startend_percent": (start_percent, end_percent),
                },
            ),
        )

#LyingSigmaSampler
def Log_Sigma_Sampler_Steps(
    model,
    x,
    sigmas,
    *,
    lss_wrapped_sampler,
    lss_dishonesty_factor,
    lss_lerpfactor,
    lss_startend_percent,
    **kwargs,
):

    start_index, end_index= lss_startend_percent


    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        adjusted_sigma = sigma
        index = torch.argmin(torch.abs(sigmas.to(sigma.device) - sigma))
        if start_index <= index <= end_index:
            # Find index of nearest element
            print(f"index {index}")
            LogSigmas = lss_dishonesty_factor[index]
            adjusted_sigma = torch.lerp(torch.tensor([LogSigmas], device=sigma.device),sigma,lss_lerpfactor)
            print(f"lss_dishonesty_factor[index] {lss_dishonesty_factor[index]}, sigma {sigma}, siggma_float {sigma_float}")
        return model(x, adjusted_sigma, **extra_args)


    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return lss_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **lss_wrapped_sampler.extra_options,
    )

class LogSigmaStepSamplerNode:
    CATEGORY = "TBG/Takeaways"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "LogSigmas" : ("SIGMAS",),
                "lerpfactor": (
                    "FLOAT",
                    {
                        "default": -0.05,
                        "min": -0.999,
                        "step": 0.01,
                        "tooltip": "Multiplier for sigmas passed to the model. -0.05 means we reduce the sigma by 5%.",
                    },
                ),
            },
            "optional": {
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100}),
                "end_step": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    @classmethod
    def go(cls, sampler, LogSigmas,     lerpfactor, *, start_step=0, end_step=1):
        return (
            KSAMPLER(
                Log_Sigma_Sampler_Steps,
                extra_options={
                    "lss_wrapped_sampler": sampler,
                    "lss_dishonesty_factor": LogSigmas,
                    "lss_lerpfactor" : lerpfactor,
                    "lss_startend_percent": (start_step, end_step),
                },
            ),
        )


import random


class PromptBatchGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames_in_seconds": ("INT", {"default": 81, "min": 1, "max": 30000, "step": 1}),
                "frames_per_batch": ("INT", {"default": 81, "min": 1, "max": 120, "step": 1}),
                "text_input_1": ("STRING", {"default": "", "multiline": True}),
                "text_input_2": ("STRING", {"default": "", "multiline": True}),
                "text_input_3": ("STRING", {"default": "", "multiline": True}),
                "text_input_4": ("STRING", {"default": "", "multiline": True}),
                "text_input_5": ("STRING", {"default": "", "multiline": True}),
                "strength_1": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "strength_2": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "strength_3": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "strength_4": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "strength_5": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_string",)
    FUNCTION = "generate_prompt_batches"
    CATEGORY = "TBG/Takeaways"
    DESCRIPTION = "Generates batches of prompts based on text inputs and their strength values"

    def generate_prompt_batches(self, total_frames_in_seconds, frames_per_batch,
                                text_input_1, text_input_2, text_input_3, text_input_4, text_input_5,
                                strength_1, strength_2, strength_3, strength_4, strength_5):


        # Calculate total batches
        total_batches = total_frames_in_seconds // frames_per_batch
        if total_frames_in_seconds % frames_per_batch > 0:
            total_batches += 1

        # Collect active text inputs (non-empty) with their strengths
        text_inputs = []
        if text_input_1.strip():
            text_inputs.append((text_input_1.strip(), strength_1))
        if text_input_2.strip():
            text_inputs.append((text_input_2.strip(), strength_2))
        if text_input_3.strip():
            text_inputs.append((text_input_3.strip(), strength_3))
        if text_input_4.strip():
            text_inputs.append((text_input_4.strip(), strength_4))
        if text_input_5.strip():
            text_inputs.append((text_input_5.strip(), strength_5))

        if not text_inputs:
            return ("",)  # Return empty string if no inputs

        # Create weighted list based on strength values
        weighted_prompts = []
        for text, strength in text_inputs:
            weighted_prompts.extend([text] * strength)

        # Generate batch strings
        batch_strings = []
        for batch_num in range(total_batches):
            # Randomly select a prompt for this batch based on weights
            selected_prompt = random.choice(weighted_prompts)
            batch_strings.append(selected_prompt)

        # Join all batches with pipe separator
        final_prompt_string = " | ".join(batch_strings)

        return (final_prompt_string,)






class VAEDecodeColorFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image with accurate colors.",)
    FUNCTION = "decode"
    CATEGORY = "TBG/Takeaways"
    DESCRIPTION = "Fast Flux VAE decode with accurate colors using optimized tiled processing."

    def decode(self, vae, samples):
        tile_size = 256
        fast_mode = True
        compression = vae.spacial_compression_decode()
        tile_latent = tile_size // compression
        overlap = tile_latent // 4  # 25% overlap for smooth blending

        if fast_mode:
            # Single-pass tiled decode (3x faster)
            images = self.decode_single_pass(vae, samples["samples"], tile_latent, overlap)
        else:
            # Use the original 3-pass tiled decode for maximum quality
            images = vae.decode_tiled(samples["samples"], tile_x=tile_latent, tile_y=tile_latent, overlap=overlap)

        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        return (images,)

    def decode_single_pass(self, vae, samples, tile_x, overlap):
        """Single-pass tiled decode - 3x faster than the original 3-pass method"""
        import comfy.utils
        from comfy.utils import ProgressBar

        vae.throw_exception_if_invalid()

        # Calculate steps for progress bar
        steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_x, overlap)
        #pbar = ProgressBar(steps)

        # Load VAE to GPU
        memory_used = vae.memory_used_decode(samples.shape, vae.vae_dtype)
        model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

        # Decode function
        decode_fn = lambda a: vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).float()

        # Single tiled_scale pass (instead of 3 passes)
        output = comfy.utils.tiled_scale(
            samples,
            decode_fn,
            tile_x,
            tile_x,  # Use square tiles
            overlap,
            upscale_amount=vae.upscale_ratio,
            output_device=vae.output_device,
            #pbar=pbar
        )

        # Apply process_output and move channels
        output = vae.process_output(output)
        return output.movedim(1, -1)

class TBG_Preview_Sender_WebSocked:
    """
    Sends preview images via ComfyUI's built-in WebSocket
    No external server needed!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "node_name": ("STRING", {"default": "refiner_preview"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "send_preview"
    OUTPUT_NODE = True
    CATEGORY = "TBG/Takeaways"

    def send_preview(self, images, node_name="refiner_preview"):
        """
        Send preview via ComfyUI's WebSocket to all connected clients
        """
        server = PromptServer.instance

        # Process first image
        if len(images) > 0:
            image = images[0]

            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Send via ComfyUI's WebSocket with custom event type
            server.send_sync("tbg_preview", {
                "node_name": node_name,
                "image": f"data:image/png;base64,{img_base64}",
                "timestamp": str(np.datetime64('now'))
            })

            print(f"[TBG] Sent preview via ComfyUI WebSocket: {node_name}")

        # Pass through images unchanged
        return (images,)



def create_hex_cone_map(width=1024, height=1024, hex_radius=32, outer_fraction=0.5,
                        inner_fraction=0.1, curve_power=0.35):
    r = float(hex_radius)
    outer_radius = r * float(outer_fraction)
    inner_radius = r * float(inner_fraction)

    dx = 1.5 * r
    dy = np.sqrt(3) * r

    # Create image with proper width x height dimensions
    image = np.ones((height, width), dtype=np.float32)
    centers = []
    y = 0.0
    row = 0
    while y - r <= height:
        x_offset = 0.0 if (row % 2 == 0) else dx / 2.0
        x = x_offset
        while x - r <= width:
            centers.append((x, y))
            x += dx
        row += 1
        y += dy

    for cx, cy in centers:
        x0 = max(int(cx - outer_radius), 0)
        x1 = min(int(cx + outer_radius) + 1, width)
        y0 = max(int(cy - outer_radius), 0)
        y1 = min(int(cy + outer_radius) + 1, height)

        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)  # ✅ Fixed **2

        t = (dist - inner_radius) / (outer_radius - inner_radius + 1e-12)
        t = np.clip(t, 0, 1)

        gradient = t ** curve_power
        gradient[dist <= inner_radius] = 0
        gradient[dist >= outer_radius] = 1

        image[y0:y1, x0:x1] = np.minimum(image[y0:y1, x0:x1], gradient)

    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


class HexConeDenoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "hex_radius": ("INT", {"default": 32, "min": 4, "max": 128}),
                "outer_fraction": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0}),
                "inner_fraction": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.4}),
                "curve_power": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 1.0})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate"
    CATEGORY = "TBG/Takeaways"

    def generate(self, input_image, hex_radius, outer_fraction, inner_fraction, curve_power):
        # ComfyUI format is BHWC (Batch, Height, Width, Channels)
        _, h, w, _ = input_image.shape

        # Generate hex cone mask for the full image dimensions
        mask_array = create_hex_cone_map(w, h, hex_radius, outer_fraction, inner_fraction, curve_power)

        # Convert to tensor (H, W) -> add batch dimension -> (1, H, W)
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0) / 255.0

        # Convert mask to IMAGE format (B, H, W, C)
        # Add channel dimension and repeat for RGB: (1, H, W) -> (1, H, W, 1) -> (1, H, W, 3)
        image_tensor = mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Return both the visual representation and the mask
        return (image_tensor, mask_tensor)


class FLUX2JSONPromptGenerator:
    """
    Enhanced FLUX.2 JSON Prompt Generator with professional presets and tooltips.
    Supports all FLUX.2 JSON schema parameters with comprehensive examples.
    """

    # ==================== CAMERA PRESETS ====================

    CAMERA_PRESETS = [
        "None / Manual Configuration",
        "Sony A7IV - 85mm f/5.6 ISO200 (Product/Portrait)",
        "Hasselblad X2D - 80mm f/2.8 ISO100 (High-end Fashion)",
        "Canon EOS R5 - 24-70mm f/4 ISO400 (Versatile Pro)",
        "Nikon Z9 - 70-200mm f/2.8 ISO800 (Sports/Action)",
        "Leica M11 - 50mm f/2 ISO100 (Street Photography)",
        "Phase One XF IQ4 - 80mm f/5.6 ISO100 (Commercial Studio)",
        "Fujifilm GFX 100 II - 110mm f/2 ISO200 (Fashion Editorial)",
        "Hasselblad 907X - 90mm f/3.5 ISO64 (Fine Art)",
        "ARRI Alexa Mini LF - 35mm f/2.8 ISO800 (Cinematic)",
        "RED Komodo 6K - 50mm f/1.8 ISO1600 (Film Production)",
        "Kodak Ektachrome 64 - 35mm f/5.6 ISO64 (Expired Film Look)",
        "Kodak Portra 400 - 50mm f/2.8 ISO400 (Film Photography)",
        "Polaroid SX-70 - Built-in f/8 ISO160 (Instant Film)",
        "2000s Canon PowerShot - 28mm f/2.8 ISO200 (Digicam Aesthetic)",
        "iPhone 15 Pro Max - 24mm f/1.78 ISO80 (Mobile Photography)",
        "DJI Inspire 3 - 24mm f/2.8 ISO400 (Aerial Photography)",
        "GoPro Hero 12 - 16mm f/2.8 ISO400 (Action POV)",
        "Macro Lens Setup - 105mm f/2.8 ISO200 (Macro Photography)",
    ]

    # ==================== CAMERA MODELS ====================

    CAMERA_MODELS = [
        "",
        "None",
        "Sony A7IV", "Sony A7R V", "Sony A1", "Sony A9 III",
        "Canon EOS R5", "Canon EOS R6 Mark II", "Canon 5D Mark IV", "Canon EOS R3",
        "Nikon Z9", "Nikon Z8", "Nikon D850", "Nikon Z6 III",
        "Fujifilm X-T5", "Fujifilm GFX 100 II", "Fujifilm X-H2S",
        "Hasselblad X2D", "Hasselblad 907X", "Hasselblad H6D-100c",
        "Leica M11", "Leica Q3", "Leica SL3",
        "Phase One XF IQ4", "Pentax 645Z",
        "Panasonic Lumix S5 II", "Panasonic GH6", "Olympus OM-1",
        "RED Komodo 6K", "ARRI Alexa Mini LF", "Blackmagic Pocket 6K",
        "iPhone 15 Pro Max", "Google Pixel 8 Pro", "Samsung Galaxy S24 Ultra",
        "DJI Inspire 3", "GoPro Hero 12",
        "Polaroid SX-70", "Kodak Ektar H35", "Lomography LC-A+",
        "Custom"
    ]

    # ==================== CAMERA ANGLES ====================

    CAMERA_ANGLES = [
        "",
        "None",
        "Eye level (neutral perspective)",
        "High angle (looking down)",
        "Low angle (looking up, heroic)",
        "Bird's eye view (directly overhead)",
        "Worm's eye view (ground level looking up)",
        "Dutch angle / Tilted (dynamic tension)",
        "Over the shoulder (conversational)",
        "Point of view / POV (first person)",
        "Aerial view (elevated perspective)",
        "Ground level (intimate low angle)",
        "Three-quarter view (classic portrait)",
        "Profile view (side perspective)",
        "Slightly elevated (editorial standard)",
        "Custom"
    ]

    # ==================== CAMERA DISTANCES ====================

    CAMERA_DISTANCES = [
        "",
        "None",
        "Extreme close-up (detail focus)",
        "Close-up (face/product detail)",
        "Medium close-up (head and shoulders)",
        "Medium shot (waist up)",
        "Medium full shot (knee level)",
        "Full shot (entire subject)",
        "Wide shot (subject in environment)",
        "Extreme wide shot (landscape)",
        "Establishing shot (scene setter)",
        "Long shot (distant perspective)",
        "Macro (extreme detail)",
        "Intimate distance (personal space)",
        "Custom"
    ]

    # ==================== FOCUS TYPES ====================

    FOCUS_TYPES = [
        "",
        "None",
        "Sharp focus throughout (everything crisp)",
        "Shallow depth of field (background blur)",
        "Deep depth of field (everything sharp)",
        "Soft focus (dreamy romantic)",
        "Rack focus (shift focus subject)",
        "Tilt-shift focus (miniature effect)",
        "Bokeh background (aesthetic blur)",
        "Subject in focus, background blur (portrait)",
        "Foreground blur, subject sharp (depth)",
        "Split focus (multiple focus points)",
        "Selective focus (spotlight effect)",
        "Sharp focus on steam rising from coffee and mug details",
        "Custom"
    ]

    # ==================== F-NUMBERS ====================

    F_NUMBERS = [
        "",
        "None",
        "f/1.2 (ultra shallow DOF)",
        "f/1.4 (very shallow DOF)",
        "f/1.8 (portrait standard)",
        "f/2.0 (low light portrait)",
        "f/2.8 (versatile shallow)",
        "f/4.0 (balanced DOF)",
        "f/5.6 (product standard)",
        "f/8.0 (landscape standard)",
        "f/11 (deep DOF)",
        "f/16 (architectural)",
        "f/22 (maximum DOF)",
        "f/32 (extreme DOF)",
        "Custom"
    ]

    # ==================== ISO VALUES ====================

    ISO_VALUES = [
        "",
        "None",
        "50 (studio perfect light)",
        "64 (medium format base)",
        "100 (bright daylight)",
        "200 (studio/outdoors)",
        "400 (versatile general)",
        "800 (low light capable)",
        "1600 (night/indoor)",
        "2000 (extreme low light)",
        "Custom"
    ]

    # ==================== LENS TYPES ====================

    LENS_TYPES = [
        "",
        "None",
        "14mm ultra wide angle",
        "16-35mm wide zoom",
        "24mm wide angle",
        "24-70mm at 35mm (standard zoom)",
        "35mm spherical lens",
        "50mm prime (normal)",
        "85mm portrait lens",
        "105mm macro",
        "135mm portrait",
        "70-200mm telephoto",
        "100-400mm super telephoto",
        "8mm fisheye",
        "Tilt-shift lens",
        "Vintage lens (character)",
        "Anamorphic lens (cinematic)",
        "Custom"
    ]

    # ==================== STYLE PRESETS ====================

    STYLE_PRESETS = [
        "",
        "None / Custom",
        "Ultra-realistic product photography with commercial quality",
        "Modern digital photography - shot on Sony A7IV, clean sharp, high dynamic range",
        "Photorealistic 3D render quality with perfect lighting and surfaces",
        "Professional editorial photography - magazine quality sharp detailed",
        "Documentary photography - authentic natural candid realistic",
        "2000s digicam style - early digital camera, slight noise, flash photography, candid",
        "80s vintage photo - film grain, warm color cast, soft focus, nostalgic",
        "Analog film photography - shot on Kodak Portra 400, natural grain, organic colors",
        "Cross-processed Ektachrome 64 - expired film from 1987, extreme color shifts, cyan-magenta split, heavy grain",
        "Kodachrome 1960s - vintage saturated colors, warm tones, rich reds and blues",
        "Black and white film - Ilford HP5 Plus, classic grain, high contrast, timeless",
        "Instant film photography - Polaroid aesthetic, soft colors, square format, nostalgic",
        "Medium format film - Hasselblad 500CM, Kodak Portra 160, classic analog quality, smooth tones",
        "Cinematic photography - shot on ARRI Alexa, film-like color grading, anamorphic aesthetic",
        "Film noir style - high contrast black and white, dramatic shadows, moody atmospheric",
        "Technicolor aesthetic - vibrant saturated colors, classic Hollywood golden age",
        "Wes Anderson style - symmetrical composition, pastel color palette, centered framing",
        "Fashion editorial photography - high contrast, vogue style, dramatic lighting",
        "High fashion runway photography - dynamic movement, professional lighting, editorial quality",
        "Fashion magazine spread - high fashion editorial styling, sophisticated composition",
        "Luxury fashion photography - elegant, sophisticated, premium quality, refined aesthetic",
        "Street fashion photography - candid urban style, authentic natural lighting",
        "Commercial advertising photography - clean professional studio, product focused",
        "Luxury brand photography - high-end elegant sophisticated premium quality",
        "E-commerce product photography - white background, clean professional, web optimized",
        "Beauty product photography - soft glamorous lighting, elegant sophisticated",
        "Food photography - appetizing professional styling, shallow DOF, mouth-watering",
        "Automotive photography - sleek dynamic lighting, reflective surfaces, powerful composition",
        "Tech product photography - clean modern minimalist, precise lighting, sharp details",
        "Photorealistic 3D render - Octane render, ray-traced lighting, perfect surfaces",
        "Architectural visualization - Unreal Engine 5, photorealistic render, accurate materials",
        "Product CGI render - studio lighting, perfect surfaces, commercial quality",
        "Pixar-style 3D animation - stylized cartoon render, vibrant colors, appealing characters",
        "Cyberpunk CGI aesthetic - neon lighting, futuristic render, high-tech atmosphere",
        "Blender Cycles render - photorealistic materials, physically accurate lighting",
        "Architectural photography - Architectural Digest style, interior design, natural lighting",
        "Real estate photography - bright inviting, wide angle, HDR processed",
        "Interior design photography - styled professional, balanced lighting, magazine quality",
        "Architectural exterior - golden hour lighting, dramatic sky, professional composition",
        "Fine art photography - museum quality, artistic composition, intentional aesthetic",
        "Surrealist photography - dreamlike, ethereal, artistic, Salvador Dali inspired",
        "Minimalist photography - clean simple, negative space, zen aesthetic",
        "Abstract photography - experimental, artistic, conceptual, non-representational",
        "Conceptual art photography - thought-provoking, symbolic, artistic narrative",
        "Studio portrait photography - professional lighting, clean background, headshot quality",
        "Environmental portrait - subject in context, natural setting, storytelling",
        "Glamour photography - soft romantic lighting, elegant beautiful, refined",
        "Lifestyle photography - natural authentic, candid moments, relatable realistic",
        "Character portrait - personality focused, dramatic lighting, storytelling",
        "Street photography - candid documentary style, natural lighting, authentic urban life",
        "Documentary photography - photojournalism, authentic storytelling, reportage style",
        "National Geographic style - nature documentary photography, stunning wildlife, environmental",
        "Travel magazine photography - stunning landscape editorial, cultural authentic, wanderlust",
        "Sports photography - dynamic action, frozen motion, dramatic peak moment",
        "Concert photography - dynamic stage lighting, motion energy, live performance",
        "Magazine cover editorial - professional layout, typography integration, newsstand quality",
        "Vogue editorial style - high fashion, dramatic lighting, sophisticated composition",
        "GQ magazine style - masculine sophisticated, clean professional, editorial quality",
        "Rolling Stone photography - music editorial, dramatic portrait, iconic style",
        "Classic superhero comic - bold colors, dynamic action, halftone dots, comic book aesthetic",
        "Manga style illustration - Japanese comic aesthetic, screentone shading, dramatic angles",
        "Graphic novel style - sophisticated illustration, cinematic panels, artistic narrative",
    ]

    # ==================== LIGHTING PRESETS ====================

    LIGHTING_PRESETS = [
        "",
        "None / Custom",
        "Natural window light - soft diffused indirect illumination",
        "Golden hour sunset lighting - warm backlit, long shadows, magical glow",
        "Blue hour twilight - cool atmospheric, soft even lighting",
        "Studio three-point lighting - professional setup, key fill rim lights",
        "Soft box lighting - diffused even coverage, minimal shadows",
        "Hard directional light - strong shadows, dramatic contrast",
        "Rim lighting - backlit edge highlight, subject separation",
        "Low-key lighting - dark moody shadows, dramatic atmosphere",
        "High-key lighting - bright airy, minimal shadows, clean",
        "Overhead diffused lighting - flat even coverage, shadowless",
        "Side lighting - dramatic shadow contrast, sculptural form",
        "Butterfly lighting - beauty standard, symmetrical nose shadow",
        "Rembrandt lighting - triangular cheek highlight, classic portrait",
        "Split lighting - half lit half shadow, dramatic portrait",
        "Ring light - beauty lighting, even soft, catchlight circles",
        "Neon lighting - colorful artificial, urban nightlife atmosphere",
        "Candlelight - warm flickering, natural intimate ambiance",
        "Moonlight - cool blue, night lighting, mysterious atmosphere",
        "Overcast daylight - soft shadowless, natural even illumination",
        "Dappled forest light - filtered through leaves, natural patterns",
        "Three-point softbox setup creating soft, diffused highlights with no harsh shadows",
        "Dramatic backlighting and energy radiating outward in waves",
    ]

    # ==================== MOOD PRESETS ====================

    MOOD_PRESETS = [
        "",
        "None / Custom",
        "Clean, professional, minimalist",
        "Warm, inviting, cozy, comfortable",
        "Cool, sophisticated, modern, sleek",
        "Dramatic, intense, moody, powerful",
        "Bright, cheerful, energetic, vibrant",
        "Calm, peaceful, serene, tranquil",
        "Mysterious, dark, atmospheric, enigmatic",
        "Romantic, soft, dreamy, intimate",
        "Bold, dynamic, powerful, confident",
        "Elegant, refined, luxurious, sophisticated",
        "Rustic, authentic, organic, natural",
        "Futuristic, sleek, high-tech, modern",
        "Nostalgic, vintage, retro, timeless",
        "Playful, fun, whimsical, joyful",
        "Epic, cinematic, grand, spectacular",
        "Tense, urgent, dramatic (comic style)",
        "Victorious, hopeful, triumphant",
    ]

    # ==================== COMPOSITION PRESETS ====================

    COMPOSITION_PRESETS = [
        "",
        "None / Custom",
        "Rule of thirds (classic balanced)",
        "Center composition (symmetrical focus)",
        "Golden ratio (natural harmony)",
        "Symmetrical balance (mirror perfection)",
        "Asymmetrical balance (dynamic tension)",
        "Leading lines (directional flow)",
        "Frame within frame (natural borders)",
        "Negative space (minimalist emphasis)",
        "Diagonal composition (dynamic energy)",
        "Pattern repetition (rhythmic design)",
        "Depth layering (foreground mid background)",
        "Foreground interest (depth anchor)",
        "Triangular composition (stable dynamic)",
        "Radial composition (outward flow)",
        "Juxtaposition (contrasting elements)",
    ]

    # ==================== SCENE EXAMPLES ====================

    SCENE_EXAMPLES = [
        "",
        "Custom scene description",
        "Professional studio product photography setup with polished concrete surface",
        "Massive computer server room with sparking circuits and red warning lights flashing on monitors",
        "Small office with computer monitors displaying code and error messages",
        "Digital cyberspace environment with floating data cubes and cascading binary code",
        "Calm server room with soft blue ambient lighting and orderly data streams flowing smoothly",
        "Urban rooftop at sunset with city skyline in background",
        "Minimalist white studio with seamless backdrop",
        "Rustic wooden table with natural window light",
        "Modern architectural interior with floor-to-ceiling windows",
        "Outdoor forest clearing with dappled sunlight through trees",
        "Industrial warehouse with dramatic overhead lighting",
        "Luxury hotel lobby with marble floors and chandeliers",
        "Cozy coffee shop interior with warm ambient lighting",
        "Fashion runway with dramatic spotlights",
        "Makeup flat lay on marble surface",
    ]

    # ==================== BACKGROUND EXAMPLES ====================

    BACKGROUND_EXAMPLES = [
        "",
        "Custom background",
        "Polished concrete surface with studio backdrop",
        "Seamless white studio background",
        "Dark gradient background fading to black",
        "Natural bokeh with soft out-of-focus lights",
        "Urban cityscape with blurred buildings",
        "Marble surface with subtle veining",
        "Wooden texture with natural grain",
        "Solid color backdrop (use color palette)",
        "Environmental context with depth",
        "Abstract gradient background",
        "Textured fabric backdrop",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_preset": (cls.CAMERA_PRESETS, {
                    "default": "Sony A7IV - 85mm f/5.6 ISO200 (Product/Portrait)",
                    "tooltip": "Camera presets with model, lens, f-number, ISO."
                }),
                "scene_preset": (cls.SCENE_EXAMPLES, {
                    "default": "Professional studio product photography setup with polished concrete surface",
                    "tooltip": "Pre-made scene descriptions or Custom."
                }),
                "scene": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Scene description. Overrides scene_preset when non-empty."
                }),
                "subject_count": ("STRING", {
                    "default": "1",
                    "tooltip": "Number of subjects (0-10). Parsed as int; invalid values default to 1."
                }),
                "subject_1_description": ("STRING", {
                    "default": "Minimalist ceramic coffee mug with steam rising from hot coffee inside",
                    "multiline": True,
                    "tooltip": "Subject 1 description."
                }),
                "subject_1_position": ("STRING", {
                    "default": "Center foreground on polished concrete surface",
                    "multiline": False,
                    "tooltip": "Subject 1 position in frame."
                }),
                "subject_1_action": ("STRING", {
                    "default": "Stationary on surface",
                    "multiline": False,
                    "tooltip": "Subject 1 action."
                }),
                "subject_1_pose": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subject 1 pose (optional)."
                }),
                "subject_1_color_palette": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subject 1 colors (#hex, comma-separated)."
                }),
                "style_preset": (cls.STYLE_PRESETS, {
                    "default": "Ultra-realistic product photography with commercial quality",
                    "tooltip": "Style preset."
                }),
                "style_text_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom style text. Overrides style_preset when non-empty."
                }),
                "lighting_preset": (cls.LIGHTING_PRESETS, {
                    "default": "Studio three-point lighting - professional setup, key fill rim lights",
                    "tooltip": "Lighting preset."
                }),
                "lighting_text_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom lighting. Overrides lighting_preset."
                }),
                "mood_preset": (cls.MOOD_PRESETS, {
                    "default": "Clean, professional, minimalist",
                    "tooltip": "Mood preset."
                }),
                "mood_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom mood. Overrides mood_preset."
                }),
                "background_preset": (cls.BACKGROUND_EXAMPLES, {
                    "default": "Polished concrete surface with studio backdrop",
                    "tooltip": "Background preset."
                }),
                "background": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Background description. Overrides background_preset."
                }),
                "composition_preset": (cls.COMPOSITION_PRESETS, {
                    "default": "Rule of thirds (classic balanced)",
                    "tooltip": "Composition preset."
                }),
                "composition_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom composition. Overrides preset."
                }),
                "camera_model_preset": (cls.CAMERA_MODELS, {
                    "default": "Sony A7IV",
                    "tooltip": "Camera model preset."
                }),
                "camera_model_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom camera model. Overrides preset."
                }),
                "camera_angle_preset": (cls.CAMERA_ANGLES, {
                    "default": "High angle (looking down)",
                    "tooltip": "Camera angle preset."
                }),
                "camera_angle_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom camera angle. Overrides preset."
                }),
                "camera_distance_preset": (cls.CAMERA_DISTANCES, {
                    "default": "Medium shot (waist up)",
                    "tooltip": "Camera distance preset."
                }),
                "camera_distance_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom camera distance. Overrides preset."
                }),
                "focus_preset": (cls.FOCUS_TYPES, {
                    "default": "Sharp focus on steam rising from coffee and mug details",
                    "tooltip": "Focus preset."
                }),
                "focus_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom focus. Overrides preset."
                }),
                "lens_preset": (cls.LENS_TYPES, {
                    "default": "85mm portrait lens",
                    "tooltip": "Lens preset."
                }),
                "lens_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom lens. Overrides preset."
                }),
                "f_number_preset": (cls.F_NUMBERS, {
                    "default": "f/5.6 (product standard)",
                    "tooltip": "Aperture preset."
                }),
                "f_number_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom f-number. Overrides preset."
                }),
                "iso_preset": (cls.ISO_VALUES, {
                    "default": "200 (studio/outdoors)",
                    "tooltip": "ISO preset."
                }),
                "iso_text_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom ISO. Overrides preset."
                }),
                "lens_mm": ("STRING", {
                    "default": "0",
                    "tooltip": "Lens focal length in mm. 0 or empty = use preset or omit."
                }),
                "shutter_speed": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Shutter speed (optional), e.g. 1/125."
                }),
                "color_hex_1": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Primary hex color (e.g. #FF5733)."
                }),
                "color_hex_2": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Secondary hex color."
                }),
                "color_hex_3": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Tertiary hex color."
                }),
                "color_hex_4": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Accent hex color 1."
                }),
                "color_hex_5": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Accent hex color 2."
                }),
                "include_shot_on_prefix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add 'Shot on [camera]' prefix to formatted prompt."
                }),
                "enable_prompt_expansion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reserved for future auto-expansion (no-op now)."
                }),
            },
            "optional": {
                "subject_2_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Subject 2 description (optional)."
                }),
                "subject_2_position": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subject 2 position."
                }),
                "subject_2_action": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subject 2 action."
                }),
                "subject_2_pose": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subject 2 pose."
                }),
                "subject_2_color_palette": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Subject 2 colors."
                }),
                "subject_3_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Subject 3 description."
                }),
                "subject_3_position": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 3 position"}),
                "subject_3_action": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 3 action"}),
                "subject_3_pose": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 3 pose"}),
                "subject_3_color_palette": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 3 colors"}),
                "subject_4_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Subject 4 description."
                }),
                "subject_4_position": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 4 position"}),
                "subject_4_action": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 4 action"}),
                "subject_4_pose": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 4 pose"}),
                "subject_4_color_palette": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 4 colors"}),
                "subject_5_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Subject 5 description."
                }),
                "subject_5_position": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 5 position"}),
                "subject_5_action": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 5 action"}),
                "subject_5_pose": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 5 pose"}),
                "subject_5_color_palette": ("STRING", {"default": "", "multiline": False, "tooltip": "Subject 5 colors"}),
                "additional_json_fields": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Extra JSON object merged into final output (advanced)."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("json_prompt", "formatted_prompt", "camera_settings", "json_to_clip_prompt")
    FUNCTION = "generate_json_prompt"
    CATEGORY = "TBG/Takeaways"
    OUTPUT_NODE = False

    # ===== Helpers =====

    def _apply_camera_preset(self, preset: str) -> Dict[str, Any]:
        settings = {"camera_model": "", "lens_mm": 0, "f_number": "", "iso": ""}
        if preset == "None / Manual Configuration":
            return settings
        parts = preset.split(" - ")
        if len(parts) < 2:
            return settings
        settings["camera_model"] = parts[0].strip()
        spec_part = parts[1].split("(")[0].strip()
        tokens = spec_part.split()
        for token in tokens:
            if "mm" in token:
                try:
                    settings["lens_mm"] = int(token.replace("mm", ""))
                except Exception:
                    pass
            elif token.startswith("f/"):
                settings["f_number"] = token
            elif token.startswith("ISO"):
                settings["iso"] = token.replace("ISO", "")
        return settings

    def _get_value_or_override(
        self,
        preset_value: str,
        text_override: str,
        none_values: tuple = ("", "None", "None / Custom", "Custom", "Custom scene description", "Custom background")
    ) -> str:
        if text_override and str(text_override).strip():
            return str(text_override).strip()
        if preset_value in none_values or preset_value is None:
            return ""
        return str(preset_value)

    def _parse_color_palette_string(self, palette_str: str) -> list:
        if not palette_str or not str(palette_str).strip():
            return []
        colors = []
        for color in str(palette_str).split(","):
            color = color.strip()
            if color:
                if not color.startswith("#"):
                    color = "#" + color
                colors.append(color)
        return colors

    def _build_subjects_array(self, subject_count: int, **kwargs) -> list:
        subjects = []
        for i in range(1, subject_count + 1):
            desc_key = f"subject_{i}_description"
            pos_key = f"subject_{i}_position"
            action_key = f"subject_{i}_action"
            pose_key = f"subject_{i}_pose"
            color_key = f"subject_{i}_color_palette"
            description = str(kwargs.get(desc_key, "")).strip()
            if not description:
                continue
            subject = {"description": description}
            position = str(kwargs.get(pos_key, "")).strip()
            if position:
                subject["position"] = position
            action = str(kwargs.get(action_key, "")).strip()
            if action:
                subject["action"] = action
            pose = str(kwargs.get(pose_key, "")).strip()
            if pose:
                subject["pose"] = pose
            color_palette = self._parse_color_palette_string(kwargs.get(color_key, ""))
            if color_palette:
                subject["color_palette"] = color_palette
            subjects.append(subject)
        return subjects

    def _build_color_palette(self, **kwargs) -> list:
        colors = []
        for i in range(1, 6):
            key = f"color_hex_{i}"
            color = str(kwargs.get(key, "")).strip()
            if color:
                if not color.startswith("#"):
                    color = "#" + color
                colors.append(color)
        return colors

    def _safe_int(self, value, default: int, min_val: int = None, max_val: int = None) -> int:
        try:
            i = int(value)
        except (TypeError, ValueError):
            i = default
        if min_val is not None and i < min_val:
            i = min_val
        if max_val is not None and i > max_val:
            i = max_val
        return i

    def _build_camera_dict(self, preset_settings: Dict[str, Any], **kwargs) -> dict:
        camera: Dict[str, Any] = {}
        angle = self._get_value_or_override(
            kwargs.get("camera_angle_preset", "None"),
            kwargs.get("camera_angle_text_override", "")
        )
        if angle and "(" in angle:
            angle = angle.split("(")[0].strip()
        if angle:
            camera["angle"] = angle
        distance = self._get_value_or_override(
            kwargs.get("camera_distance_preset", "None"),
            kwargs.get("camera_distance_text_override", "")
        )
        if distance and "(" in distance:
            distance = distance.split("(")[0].strip()
        if distance:
            camera["distance"] = distance
        focus = self._get_value_or_override(
            kwargs.get("focus_preset", "None"),
            kwargs.get("focus_text_override", "")
        )
        if focus and "(" in focus:
            focus = focus.split("(")[0].strip()
        if focus:
            camera["focus"] = focus
        lens = self._get_value_or_override(
            kwargs.get("lens_preset", "None"),
            kwargs.get("lens_text_override", "")
        )
        if lens and "(" in lens:
            lens = lens.split("(")[0].strip()
        if lens:
            camera["lens"] = lens
        lens_mm = kwargs.get("lens_mm", 0)
        lens_mm = self._safe_int(lens_mm, preset_settings.get("lens_mm", 0), 0, 600)
        if lens_mm > 0:
            camera["lens-mm"] = lens_mm
        f_number = self._get_value_or_override(
            kwargs.get("f_number_preset", "None"),
            kwargs.get("f_number_text_override", "")
        )
        if not f_number and preset_settings.get("f_number"):
            f_number = preset_settings["f_number"]
        if f_number and "(" in f_number:
            f_number = f_number.split("(")[0].strip()
        if f_number:
            camera["f-number"] = f_number
        iso = self._get_value_or_override(
            kwargs.get("iso_preset", "None"),
            kwargs.get("iso_text_override", "")
        )
        if not iso and preset_settings.get("iso"):
            iso = preset_settings["iso"]
        if iso and "(" in iso:
            iso = iso.split("(")[0].strip()
        if iso:
            camera["ISO"] = int(iso) if str(iso).isdigit() else iso
        shutter = str(kwargs.get("shutter_speed", "")).strip()
        if shutter:
            camera["shutter_speed"] = shutter
        return camera

    def _build_clip_friendly_prompt_from_dict(self, data: Dict[str, Any]) -> str:
        parts: list[str] = []
        scene = data.get("scene")
        if isinstance(scene, str) and scene.strip():
            parts.append(scene.strip())
        subjects = data.get("subjects", [])
        if isinstance(subjects, list) and subjects:
            for idx, s in enumerate(subjects, 1):
                if not isinstance(s, dict):
                    continue
                desc = s.get("description", "")
                pos = s.get("position", "")
                action = s.get("action", "") or s.get("pose", "")
                colors = s.get("color_palette") or s.get("colors")
                subject_fragments = []
                if desc:
                    subject_fragments.append(desc)
                if pos:
                    subject_fragments.append(f"positioned {pos}")
                if action:
                    subject_fragments.append(action)
                if colors and isinstance(colors, list) and colors:
                    subject_fragments.append(
                        "colors " + ", ".join(str(c) for c in colors if c)
                    )
                if subject_fragments:
                    parts.append(f"Subject {idx}: " + ", ".join(subject_fragments))
        style = data.get("style")
        if isinstance(style, str) and style.strip():
            parts.append(style.strip())
        color_palette = data.get("color_palette") or data.get("color_scheme")
        if isinstance(color_palette, list) and color_palette:
            colors_text = ", ".join(str(c) for c in color_palette if c)
            if colors_text:
                parts.append(f"Color palette: {colors_text}")
        lighting = data.get("lighting")
        if isinstance(lighting, str) and lighting.strip():
            parts.append("Lighting: " + lighting.strip())
        mood = data.get("mood")
        if isinstance(mood, str) and mood.strip():
            parts.append("Mood: " + mood.strip())
        background = data.get("background")
        if isinstance(background, str) and background.strip():
            parts.append("Background: " + background.strip())
        composition = data.get("composition")
        if isinstance(composition, str) and composition.strip():
            parts.append("Composition: " + composition.split("(")[0].strip())
        camera = data.get("camera")
        if isinstance(camera, dict):
            cam_parts = []
            cam_model = camera.get("camera_model")
            angle = camera.get("angle")
            distance = camera.get("distance")
            focus = camera.get("focus") or camera.get("depth_of_field")
            lens = camera.get("lens")
            lens_mm = camera.get("lens-mm")
            fnum = camera.get("f-number")
            iso = camera.get("ISO")
            if cam_model:
                cam_parts.append(cam_model)
            if angle:
                cam_parts.append(angle)
            if distance:
                cam_parts.append(distance)
            if focus:
                cam_parts.append(focus)
            if lens:
                cam_parts.append(lens)
            if lens_mm:
                cam_parts.append(f"{lens_mm}mm")
            if fnum:
                cam_parts.append(fnum)
            if iso:
                cam_parts.append(f"ISO {iso}")
            if cam_parts:
                parts.append("Camera: " + ", ".join(str(c) for c in cam_parts if c))
        if not parts:
            return ""
        prompt = ". ".join(parts)
        if not prompt.endswith("."):
            prompt += "."
        return prompt

    # ===== Main =====

    def generate_json_prompt(
        self,
        camera_preset: str, scene_preset: str, scene: str,
        subject_count: str, subject_1_description: str,
        subject_1_position: str, subject_1_action: str,
        subject_1_pose: str, subject_1_color_palette: str,
        style_preset: str, style_text_override: str,
        lighting_preset: str, lighting_text_override: str,
        mood_preset: str, mood_text_override: str,
        background_preset: str, background: str,
        composition_preset: str, composition_text_override: str,
        camera_model_preset: str, camera_model_text_override: str,
        camera_angle_preset: str, camera_angle_text_override: str,
        camera_distance_preset: str, camera_distance_text_override: str,
        focus_preset: str, focus_text_override: str,
        lens_preset: str, lens_text_override: str,
        f_number_preset: str, f_number_text_override: str,
        iso_preset: str, iso_text_override: str,
        lens_mm: str, shutter_speed: str,
        color_hex_1: str, color_hex_2: str, color_hex_3: str,
        color_hex_4: str, color_hex_5: str,
        include_shot_on_prefix: bool, enable_prompt_expansion: bool,
        **optional_kwargs
    ) -> Tuple[str, str, str, str]:

        try:
            preset_settings = self._apply_camera_preset(camera_preset)

            # Apply camera presets as defaults when per-field controls are unset
            if not str(lens_mm).strip() or str(lens_mm).strip() == "0":
                if preset_settings.get("lens_mm", 0) > 0:
                    lens_mm = str(preset_settings["lens_mm"])
            if not f_number_text_override or not str(f_number_text_override).strip():
                if (not f_number_preset) or f_number_preset in ("", "None"):
                    if preset_settings.get("f_number"):
                        f_number_preset = preset_settings["f_number"]
            if not iso_text_override or not str(iso_text_override).strip():
                if (not iso_preset) or iso_preset in ("", "None"):
                    if preset_settings.get("iso"):
                        iso_preset = str(preset_settings["iso"])

            # Resolve active camera_model once
            camera_model = str(camera_model_text_override).strip() if camera_model_text_override else ""
            if not camera_model and preset_settings.get("camera_model"):
                camera_model = preset_settings["camera_model"]
            elif not camera_model and camera_model_preset not in ("", "None", None):
                camera_model = camera_model_preset

            subject_count_int = self._safe_int(subject_count, 1, 0, 10)
            lens_mm_int = self._safe_int(lens_mm, preset_settings.get("lens_mm", 0), 0, 600)

            prompt_dict: Dict[str, Any] = {}

            # Scene
            scene_text = str(scene).strip()
            if not scene_text and scene_preset not in ("", "Custom scene description", None):
                scene_text = scene_preset
            if scene_text:
                prompt_dict["scene"] = scene_text

            # Subjects
            all_kwargs = {
                "subject_1_description": subject_1_description,
                "subject_1_position": subject_1_position,
                "subject_1_action": subject_1_action,
                "subject_1_pose": subject_1_pose,
                "subject_1_color_palette": subject_1_color_palette,
                **optional_kwargs
            }
            subjects = self._build_subjects_array(subject_count_int, **all_kwargs)
            if subjects:
                prompt_dict["subjects"] = subjects

            # Style
            style = self._get_value_or_override(style_preset, style_text_override)
            if style:
                prompt_dict["style"] = style

            # Palette
            colors = self._build_color_palette(
                color_hex_1=color_hex_1,
                color_hex_2=color_hex_2,
                color_hex_3=color_hex_3,
                color_hex_4=color_hex_4,
                color_hex_5=color_hex_5
            )
            if colors:
                prompt_dict["color_palette"] = colors

            # Lighting
            lighting = self._get_value_or_override(lighting_preset, lighting_text_override)
            if lighting and "(" in lighting:
                lighting = lighting.split("(")[0].strip()
            if lighting:
                prompt_dict["lighting"] = lighting

            # Mood
            mood = self._get_value_or_override(mood_preset, mood_text_override)
            if mood:
                prompt_dict["mood"] = mood

            # Background
            background_text = str(background).strip()
            if not background_text and background_preset not in ("", "Custom background", None):
                background_text = background_preset
            if background_text:
                prompt_dict["background"] = background_text

            # Composition
            composition = self._get_value_or_override(composition_preset, composition_text_override)
            if composition and "(" in composition:
                composition = composition.split("(")[0].strip()
            if composition:
                prompt_dict["composition"] = composition

            # Camera dict
            camera_kwargs = {
                "camera_angle_preset": camera_angle_preset,
                "camera_angle_text_override": camera_angle_text_override,
                "camera_distance_preset": camera_distance_preset,
                "camera_distance_text_override": camera_distance_text_override,
                "focus_preset": focus_preset,
                "focus_text_override": focus_text_override,
                "lens_preset": lens_preset,
                "lens_text_override": lens_text_override,
                "f_number_preset": f_number_preset,
                "f_number_text_override": f_number_text_override,
                "iso_preset": iso_preset,
                "iso_text_override": iso_text_override,
                "lens_mm": lens_mm_int,
                "shutter_speed": shutter_speed
            }
            camera = self._build_camera_dict(preset_settings, **camera_kwargs)
            if camera:
                prompt_dict["camera"] = camera

            # Ensure camera_model is stored in JSON under camera
            if camera_model:
                if "camera" not in prompt_dict:
                    prompt_dict["camera"] = {}
                prompt_dict["camera"]["camera_model"] = camera_model

            # Additional JSON fields
            additional_json = str(optional_kwargs.get("additional_json_fields", "")).strip()
            if additional_json:
                try:
                    extra = json.loads(additional_json)
                    if isinstance(extra, dict):
                        prompt_dict.update(extra)
                except json.JSONDecodeError as e:
                    print(f"[FLUX2JSONPromptGenerator] Warning: cannot parse additional_json_fields: {e}")

            json_string = json.dumps(prompt_dict, indent=2, ensure_ascii=False)

            # Build formatted_prompt
            formatted_parts = []
            if camera_model and include_shot_on_prefix:
                formatted_parts.append(f"Shot on {camera_model}")
            if style:
                formatted_parts.append(style)
            if scene_text:
                formatted_parts.append(scene_text)
            for subject in subjects:
                s_parts = [subject.get("description", "")]
                if subject.get("position"):
                    s_parts.append(f"positioned at {subject['position']}")
                if subject.get("action"):
                    s_parts.append(subject["action"])
                if subject.get("pose"):
                    s_parts.append(subject["pose"])
                formatted_parts.append(", ".join([p for p in s_parts if p]))
            if lighting:
                formatted_parts.append(f"Lighting: {lighting}")
            if mood:
                formatted_parts.append(f"Mood: {mood}")
            if background_text:
                formatted_parts.append(f"Background: {background_text}")
            if composition:
                formatted_parts.append(f"Composition: {composition}")
            if colors:
                formatted_parts.append(f"Color palette: {', '.join(colors)}")
            formatted_prompt = ". ".join(formatted_parts)
            if formatted_prompt and not formatted_prompt.endswith("."):
                formatted_prompt += "."

            # Camera settings summary
            camera_summary_parts = []
            if camera_model:
                camera_summary_parts.append(f"Camera: {camera_model}")
            if camera.get("lens"):
                camera_summary_parts.append(f"Lens: {camera['lens']}")
            if camera.get("lens-mm"):
                camera_summary_parts.append(f"{camera['lens-mm']}mm")
            if camera.get("f-number"):
                camera_summary_parts.append(camera["f-number"])
            if camera.get("ISO"):
                camera_summary_parts.append(f"ISO {camera['ISO']}")
            if camera.get("shutter_speed"):
                camera_summary_parts.append(camera["shutter_speed"])
            if camera.get("angle"):
                camera_summary_parts.append(f"Angle: {camera['angle']}")
            if camera.get("distance"):
                camera_summary_parts.append(f"Distance: {camera['distance']}")
            camera_settings = " | ".join(camera_summary_parts) if camera_summary_parts else "No camera settings specified"

            # CLIP-safe JSON-to-text prompt
            json_to_clip_prompt = self._build_clip_friendly_prompt_from_dict(prompt_dict)

            return (json_string, formatted_prompt, camera_settings, json_to_clip_prompt)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error generating JSON prompt: {str(e)}"
            print(f"[FLUX2JSONPromptGenerator] {error_msg}")
            return (json.dumps({"error": error_msg}), error_msg, error_msg, error_msg)


NODE_CLASS_MAPPINGS = {
    "VAEDecodeColorFix": VAEDecodeColorFix,
    "PromptBatchGenerator": PromptBatchGenerator,
    "ModelSamplingFluxGradual": ModelSamplingFluxGradual,
    "PolyExponentialSigmaAdder": PolyExponentialSigmaAdder,
    "BasicSchedulerNormalized": BasicSchedulerNormalized,
    "LogSigmaSamplerNode":LogSigmaSamplerNode,
    "LogSigmaStepSamplerNode":LogSigmaStepSamplerNode,
    "TBG_FluxKontextStabilizer":TBG_FluxKontextStabilizer,
    "TBG_Preview_Sender_WebSocked": TBG_Preview_Sender_WebSocked,
    "HexConeDenoiseMask": HexConeDenoiseMask,
    "FLUX2JSONPromptGenerator": FLUX2JSONPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeColorFix": "VAE Decode ColorFix",
    "PromptBatchGenerator": "Prompt Batch Generator",
    "ModelSamplingFluxGradual": "Model Sampling Flux Gradual",
    "PolyExponentialSigmaAdder": "PolyExponential Sigma Adder",
    "BasicSchedulerNormalized": "Basic Scheduler Normalized",
    "LogSigmaSamplerNode":"LogSigmaSamplerNode",
    "LogSigmaStepSamplerNode":"LogSigmaStepSamplerNode",
    "TBG_FluxKontextStabilizer":"TBG_FluxKontextStabilizer",
    "TBG_Preview_Sender_WebSocked": "TBG Preview Sender (WebSocket)",
    "HexConeDenoiseMask": "TBG Hex Cone Mask",
    "FLUX2JSONPromptGenerator": "FLUX.2 JSON Prompt Generator"
}

