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


"""
FLUX.2 JSON Prompt Generator Node for ComfyUI
Production-ready custom node for generating structured JSON prompts
Based on Black Forest Labs FLUX.2 Prompting Guide
"""

import json
from typing import Dict, Any, Tuple


class FLUX2JSONPromptGenerator:
    """
    Generate structured JSON prompts for FLUX.2 image generation.
    Provides extensive presets with text override capabilities.
    """

    # Camera Models - 30+ professional cameras
    CAMERA_MODELS = [
        "None",
        "Sony A7IV",
        "Sony A7R V",
        "Sony A1",
        "Canon EOS R5",
        "Canon EOS R6 Mark II",
        "Canon 5D Mark IV",
        "Nikon Z9",
        "Nikon Z8",
        "Nikon D850",
        "Fujifilm X-T5",
        "Fujifilm GFX 100 II",
        "Hasselblad X2D",
        "Hasselblad 907X",
        "Leica M11",
        "Leica Q3",
        "Phase One XF IQ4",
        "Pentax 645Z",
        "Panasonic Lumix S5 II",
        "Panasonic GH6",
        "Olympus OM-1",
        "RED Komodo 6K",
        "ARRI Alexa Mini LF",
        "Blackmagic Pocket 6K",
        "iPhone 15 Pro Max",
        "Google Pixel 8 Pro",
        "DJI Inspire 3",
        "GoPro Hero 12",
        "Polaroid SX-70",
        "Kodak Ektar H35",
        "Lomography LC-A+",
        "Custom (use text input)"
    ]

    # Camera Angles - 10+ professional angles
    CAMERA_ANGLES = [
        "None",
        "Eye level",
        "High angle",
        "Low angle",
        "Bird's eye view",
        "Worm's eye view",
        "Dutch angle / Tilted",
        "Over the shoulder",
        "Point of view (POV)",
        "Aerial view",
        "Ground level",
        "Three-quarter view",
        "Profile view",
        "Custom (use text input)"
    ]

    # Camera Distances - 10+ professional distances
    CAMERA_DISTANCES = [
        "None",
        "Extreme close-up",
        "Close-up",
        "Medium close-up",
        "Medium shot",
        "Medium full shot",
        "Full shot",
        "Wide shot",
        "Extreme wide shot",
        "Establishing shot",
        "Long shot",
        "Macro",
        "Custom (use text input)"
    ]

    # Focus Types - 10+ focus settings
    FOCUS_TYPES = [
        "None",
        "Sharp focus throughout",
        "Shallow depth of field",
        "Deep depth of field",
        "Soft focus",
        "Rack focus",
        "Tilt-shift focus",
        "Bokeh background",
        "Subject in focus, background blur",
        "Foreground blur, subject sharp",
        "Split focus",
        "Selective focus",
        "Custom (use text input)"
    ]

    # F-Numbers - All standard aperture values
    F_NUMBERS = [
        "None",
        "f/1.2", "f/1.4", "f/1.8", "f/2.0", "f/2.8", "f/4.0",
        "f/5.6", "f/8.0", "f/11", "f/16", "f/22", "f/32",
        "Custom (use text input)"
    ]

    # ISO Values - 50 to 2000
    ISO_VALUES = [
        "None",
        "50", "100", "200", "400", "800", "1600", "2000",
        "Custom (use text input)"
    ]

    # Lens Types - Professional lens options
    LENS_TYPES = [
        "None",
        "24-70mm at 35mm",
        "50mm prime",
        "85mm portrait lens",
        "35mm spherical lens",
        "24mm wide angle",
        "70-200mm telephoto",
        "14mm ultra wide",
        "105mm macro",
        "135mm portrait",
        "16-35mm wide zoom",
        "100-400mm super telephoto",
        "8mm fisheye",
        "Tilt-shift lens",
        "Vintage lens",
        "Anamorphic lens",
        "Custom (use text input)"
    ]

    # Style Presets - 20+ professional styles
    STYLE_PRESETS = [
        "None / Custom",
        # Photorealistic Styles
        "Ultra-realistic product photography with commercial quality",
        "Modern digital photography - clean sharp high dynamic range",
        "2000s digicam style - early digital camera candid flash photography",
        "80s vintage photo - film grain warm color cast soft focus",
        "Analog film photography - shot on Kodak Portra 400 natural grain",
        "Cinematic photography - shot on ARRI Alexa film-like color grading",
        "Fashion editorial photography - high contrast vogue style",
        "Street photography - candid documentary style natural lighting",
        "Studio portrait photography - professional lighting clean background",
        "Lifestyle photography - natural authentic candid moments",

        # CGI & Renders
        "Photorealistic 3D render - Octane render ray-traced lighting",
        "Architectural visualization - Unreal Engine 5 photorealistic render",
        "Product CGI render - studio lighting perfect surfaces",
        "Pixar-style 3D animation - stylized cartoon render",
        "Cyberpunk CGI aesthetic - neon lighting futuristic render",

        # Commercial & Design
        "Commercial advertising photography - clean professional studio",
        "Luxury brand photography - high-end elegant sophisticated",
        "E-commerce product photography - white background clean",
        "Food photography - appetizing professional styling",
        "Beauty product photography - soft glamorous lighting",

        # Artistic Styles
        "Fine art photography - museum quality artistic composition",
        "Surrealist photography - dreamlike ethereal artistic",
        "Minimalist photography - clean simple negative space",
        "Abstract photography - experimental artistic conceptual",
        "Documentary photography - photojournalism authentic storytelling",

        # Magazine & Editorial
        "Magazine cover editorial - professional layout typography",
        "Fashion magazine spread - high fashion editorial styling",
        "Travel magazine photography - stunning landscape editorial",
        "National Geographic style - nature documentary photography",
        "Architectural Digest style - interior design photography",

        # Vintage & Film
        "Cross-processed Ektachrome 64 - expired film color shifts",
        "Kodachrome 1960s - vintage saturated colors warm tones",
        "Black and white film - Ilford HP5 classic grain",
        "Instant film photography - Polaroid aesthetic soft colors",
        "Medium format film - Hasselblad classic analog quality"
    ]

    # Lighting Presets
    LIGHTING_PRESETS = [
        "None / Custom",
        "Natural window light - soft diffused",
        "Golden hour sunset lighting - warm backlit",
        "Studio three-point lighting - professional setup",
        "Soft box lighting - diffused even coverage",
        "Hard directional light - strong shadows dramatic",
        "Rim lighting - backlit edge highlight",
        "Low-key lighting - dark moody shadows",
        "High-key lighting - bright airy minimal shadows",
        "Overhead diffused lighting - flat even coverage",
        "Side lighting - dramatic shadow contrast",
        "Ring light - beauty lighting even soft",
        "Neon lighting - colorful artificial urban",
        "Candlelight - warm flickering natural",
        "Moonlight - cool blue night lighting",
        "Overcast daylight - soft shadowless natural"
    ]

    # Mood Presets
    MOOD_PRESETS = [
        "None / Custom",
        "Clean professional minimalist",
        "Warm inviting cozy",
        "Cool sophisticated modern",
        "Dramatic intense moody",
        "Bright cheerful energetic",
        "Calm peaceful serene",
        "Mysterious dark atmospheric",
        "Romantic soft dreamy",
        "Bold dynamic powerful",
        "Elegant refined luxurious",
        "Rustic authentic organic",
        "Futuristic sleek high-tech",
        "Nostalgic vintage retro",
        "Playful fun whimsical",
        "Epic cinematic grand"
    ]

    # Composition Presets
    COMPOSITION_PRESETS = [
        "None / Custom",
        "Rule of thirds",
        "Center composition",
        "Golden ratio",
        "Symmetrical balance",
        "Asymmetrical balance",
        "Leading lines",
        "Frame within frame",
        "Negative space",
        "Diagonal composition",
        "Pattern repetition",
        "Depth layering",
        "Foreground interest"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Scene Description
                "scene": ("STRING", {
                    "default": "Professional studio setup",
                    "multiline": True
                }),

                # Subject Configuration
                "subject_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10
                }),
                "subject_1_description": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "subject_1_position": ("STRING", {
                    "default": "Center foreground"
                }),
                "subject_1_action": ("STRING", {
                    "default": ""
                }),

                # Style Configuration
                "style_preset": (cls.STYLE_PRESETS, {
                    "default": "None / Custom"
                }),
                "style_text_override": ("STRING", {
                    "default": "",
                    "multiline": False
                }),

                # Lighting
                "lighting_preset": (cls.LIGHTING_PRESETS, {
                    "default": "None / Custom"
                }),
                "lighting_text_override": ("STRING", {
                    "default": "",
                    "multiline": False
                }),

                # Mood
                "mood_preset": (cls.MOOD_PRESETS, {
                    "default": "None / Custom"
                }),
                "mood_text_override": ("STRING", {
                    "default": ""
                }),

                # Background
                "background": ("STRING", {
                    "default": "Studio backdrop",
                    "multiline": True
                }),

                # Composition
                "composition_preset": (cls.COMPOSITION_PRESETS, {
                    "default": "None / Custom"
                }),
                "composition_text_override": ("STRING", {
                    "default": ""
                }),

                # Camera Settings
                "camera_model_preset": (cls.CAMERA_MODELS, {
                    "default": "None"
                }),
                "camera_model_text_override": ("STRING", {
                    "default": ""
                }),

                "camera_angle_preset": (cls.CAMERA_ANGLES, {
                    "default": "None"
                }),
                "camera_angle_text_override": ("STRING", {
                    "default": ""
                }),

                "camera_distance_preset": (cls.CAMERA_DISTANCES, {
                    "default": "None"
                }),
                "camera_distance_text_override": ("STRING", {
                    "default": ""
                }),

                "focus_preset": (cls.FOCUS_TYPES, {
                    "default": "None"
                }),
                "focus_text_override": ("STRING", {
                    "default": ""
                }),

                "lens_preset": (cls.LENS_TYPES, {
                    "default": "None"
                }),
                "lens_text_override": ("STRING", {
                    "default": ""
                }),

                "f_number_preset": (cls.F_NUMBERS, {
                    "default": "None"
                }),
                "f_number_text_override": ("STRING", {
                    "default": ""
                }),

                "iso_preset": (cls.ISO_VALUES, {
                    "default": "None"
                }),
                "iso_text_override": ("STRING", {
                    "default": ""
                }),

                "lens_mm": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 600
                }),

                # Color Palette
                "color_hex_1": ("STRING", {
                    "default": ""
                }),
                "color_hex_2": ("STRING", {
                    "default": ""
                }),
                "color_hex_3": ("STRING", {
                    "default": ""
                }),
                "color_hex_4": ("STRING", {
                    "default": ""
                }),
                "color_hex_5": ("STRING", {
                    "default": ""
                }),
            },
            "optional": {
                # Additional Subjects
                "subject_2_description": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "subject_2_position": ("STRING", {
                    "default": ""
                }),
                "subject_2_action": ("STRING", {
                    "default": ""
                }),
                "subject_3_description": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "subject_3_position": ("STRING", {
                    "default": ""
                }),
                "subject_3_action": ("STRING", {
                    "default": ""
                }),
                "subject_4_description": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "subject_4_position": ("STRING", {
                    "default": ""
                }),
                "subject_4_action": ("STRING", {
                    "default": ""
                }),
                "subject_5_description": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "subject_5_position": ("STRING", {
                    "default": ""
                }),
                "subject_5_action": ("STRING", {
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json_prompt", "formatted_prompt")
    FUNCTION = "generate_json_prompt"
    CATEGORY = "FLUX2/Prompt Generation"

    def _get_value_or_override(self, preset_value: str, text_override: str, none_values: list = ["None", "None / Custom"]) -> str:
        """
        Returns text override if provided, otherwise returns preset value.
        Returns empty string if preset is in none_values list.
        """
        # Text override takes priority
        if text_override and text_override.strip():
            return text_override.strip()

        # Check if preset is a "none" value
        if preset_value in none_values:
            return ""

        return preset_value

    def _build_subjects_array(self, subject_count: int, **kwargs) -> list:
        """
        Build subjects array from input parameters.
        """
        subjects = []

        for i in range(1, subject_count + 1):
            desc_key = f"subject_{i}_description"
            pos_key = f"subject_{i}_position"
            action_key = f"subject_{i}_action"

            description = kwargs.get(desc_key, "").strip()

            # Skip empty subjects
            if not description:
                continue

            subject = {
                "description": description
            }

            position = kwargs.get(pos_key, "").strip()
            if position:
                subject["position"] = position

            action = kwargs.get(action_key, "").strip()
            if action:
                subject["action"] = action

            subjects.append(subject)

        return subjects

    def _build_color_palette(self, **kwargs) -> list:
        """
        Build color palette array from hex color inputs.
        """
        colors = []

        for i in range(1, 6):
            color_key = f"color_hex_{i}"
            color = kwargs.get(color_key, "").strip()

            if color:
                # Ensure color starts with #
                if not color.startswith("#"):
                    color = "#" + color
                colors.append(color)

        return colors

    def _build_camera_dict(self, **kwargs) -> dict:
        """
        Build camera configuration dictionary.
        """
        camera = {}

        # Camera angle
        angle = self._get_value_or_override(
            kwargs.get("camera_angle_preset", "None"),
            kwargs.get("camera_angle_text_override", "")
        )
        if angle:
            camera["angle"] = angle

        # Camera distance
        distance = self._get_value_or_override(
            kwargs.get("camera_distance_preset", "None"),
            kwargs.get("camera_distance_text_override", "")
        )
        if distance:
            camera["distance"] = distance

        # Focus
        focus = self._get_value_or_override(
            kwargs.get("focus_preset", "None"),
            kwargs.get("focus_text_override", "")
        )
        if focus:
            camera["focus"] = focus

        # Lens
        lens = self._get_value_or_override(
            kwargs.get("lens_preset", "None"),
            kwargs.get("lens_text_override", "")
        )
        if lens:
            camera["lens"] = lens

        # Lens mm
        lens_mm = kwargs.get("lens_mm", 0)
        if lens_mm > 0:
            camera["lens-mm"] = lens_mm

        # F-number
        f_number = self._get_value_or_override(
            kwargs.get("f_number_preset", "None"),
            kwargs.get("f_number_text_override", "")
        )
        if f_number:
            camera["f-number"] = f_number

        # ISO
        iso = self._get_value_or_override(
            kwargs.get("iso_preset", "None"),
            kwargs.get("iso_text_override", "")
        )
        if iso:
            camera["ISO"] = int(iso) if iso.isdigit() else iso

        return camera

    def generate_json_prompt(self, scene: str, subject_count: int,
                             subject_1_description: str, subject_1_position: str,
                             subject_1_action: str, style_preset: str,
                             style_text_override: str, lighting_preset: str,
                             lighting_text_override: str, mood_preset: str,
                             mood_text_override: str, background: str,
                             composition_preset: str, composition_text_override: str,
                             camera_model_preset: str, camera_model_text_override: str,
                             camera_angle_preset: str, camera_angle_text_override: str,
                             camera_distance_preset: str, camera_distance_text_override: str,
                             focus_preset: str, focus_text_override: str,
                             lens_preset: str, lens_text_override: str,
                             f_number_preset: str, f_number_text_override: str,
                             iso_preset: str, iso_text_override: str,
                             lens_mm: int, color_hex_1: str, color_hex_2: str,
                             color_hex_3: str, color_hex_4: str, color_hex_5: str,
                             **optional_kwargs) -> Tuple[str, str]:
        """
        Generate FLUX.2 JSON prompt from inputs.

        Returns:
            Tuple of (json_string, formatted_prompt_string)
        """

        try:
            # Build the prompt dictionary
            prompt_dict = {}

            # Scene
            if scene and scene.strip():
                prompt_dict["scene"] = scene.strip()

            # Subjects
            all_kwargs = {
                "subject_1_description": subject_1_description,
                "subject_1_position": subject_1_position,
                "subject_1_action": subject_1_action,
                **optional_kwargs
            }
            subjects = self._build_subjects_array(subject_count, **all_kwargs)
            if subjects:
                prompt_dict["subjects"] = subjects

            # Style
            style = self._get_value_or_override(style_preset, style_text_override)
            if style:
                prompt_dict["style"] = style

            # Color Palette
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
            if lighting:
                prompt_dict["lighting"] = lighting

            # Mood
            mood = self._get_value_or_override(mood_preset, mood_text_override)
            if mood:
                prompt_dict["mood"] = mood

            # Background
            if background and background.strip():
                prompt_dict["background"] = background.strip()

            # Composition
            composition = self._get_value_or_override(composition_preset, composition_text_override)
            if composition:
                prompt_dict["composition"] = composition

            # Camera
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
                "lens_mm": lens_mm
            }
            camera = self._build_camera_dict(**camera_kwargs)
            if camera:
                prompt_dict["camera"] = camera

            # Generate JSON string with proper formatting
            json_string = json.dumps(prompt_dict, indent=2, ensure_ascii=False)

            # Generate formatted text prompt
            formatted_parts = []

            # Add camera model if specified
            camera_model = self._get_value_or_override(camera_model_preset, camera_model_text_override)
            if camera_model:
                formatted_parts.append(f"Shot on {camera_model}")

            # Add style
            if style:
                formatted_parts.append(style)

            # Add scene
            if scene and scene.strip():
                formatted_parts.append(scene.strip())

            # Add subjects
            for i, subject in enumerate(subjects, 1):
                subject_desc = subject.get("description", "")
                subject_pos = subject.get("position", "")
                subject_action = subject.get("action", "")

                subject_text = f"Subject {i}: {subject_desc}"
                if subject_pos:
                    subject_text += f" positioned at {subject_pos}"
                if subject_action:
                    subject_text += f", {subject_action}"
                formatted_parts.append(subject_text)

            # Add lighting
            if lighting:
                formatted_parts.append(f"Lighting: {lighting}")

            # Add mood
            if mood:
                formatted_parts.append(f"Mood: {mood}")

            # Add composition
            if composition:
                formatted_parts.append(f"Composition: {composition}")

            # Add camera details
            if camera:
                camera_details = []
                if "angle" in camera:
                    camera_details.append(camera["angle"])
                if "distance" in camera:
                    camera_details.append(camera["distance"])
                if "lens" in camera:
                    camera_details.append(camera["lens"])
                if "lens-mm" in camera:
                    camera_details.append(f"{camera['lens-mm']}mm")
                if "f-number" in camera:
                    camera_details.append(camera["f-number"])
                if "ISO" in camera:
                    camera_details.append(f"ISO {camera['ISO']}")

                if camera_details:
                    formatted_parts.append(f"Camera: {', '.join(camera_details)}")

            # Add colors
            if colors:
                formatted_parts.append(f"Colors: {', '.join(colors)}")

            formatted_prompt = ". ".join(formatted_parts) + "."

            return (json_string, formatted_prompt)

        except Exception as e:
            error_msg = f"Error generating JSON prompt: {str(e)}"
            print(f"[FLUX2JSONPromptGenerator] {error_msg}")
            return (json.dumps({"error": error_msg}), error_msg)



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

