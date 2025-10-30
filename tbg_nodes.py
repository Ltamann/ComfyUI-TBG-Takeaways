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

Part of the code includes ModelSampleFlux Normaized from 42lux:

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
import torch.nn.functional as F
import comfy
import math
from comfy.samplers import KSAMPLER
from comfy.samplers import KSAMPLER
import comfy.utils
import torch
from comfy import model_management
from server import PromptServer
import base64
from PIL import Image
import io
import numpy as np

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
    CATEGORY = "TBG"

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
}


