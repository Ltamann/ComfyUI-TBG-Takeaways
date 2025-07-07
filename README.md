# ComfyUI-TBG-Takeaways
# TGB’s ComfyUI Development Takeaways

A curated collection of reusable ComfyUI nodes developed by TGB during Patreon projects. These sidecodes encapsulate key breakthroughs in model sampling, noise scheduling, and image refinement for enhanced stable diffusion workflows.

---

## Table of Contents

- [Included Nodes](#included-nodes)  
- [Overview](#overview)  
- [License](#license)  
- [Support](#support)  

---

## Included Nodes

- **TBG_FluxKontextStabilizer** *(New!)*  
  Developed specifically for the **TBG ETUR** (Enhanced Tiled Upscaler and Refiner), this node maintains exact positioning of reference images in final outputs. It stabilizes spatial context during tiled upscaling and refinement to ensure high-fidelity alignment and image coherence.

- **ModelSamplingFluxGradual**  
  Implements gradual flux-based sampling control for smoother transitions during model sampling - ModelSamplingFluxGradual interpolates between ModelSamplingFlux and ModelSamplingFlux Normalized. This allows for the best of both approaches.
Detailed Inforamtion here: https://www.patreon.com/posts/125571636/edit

- **PolyExponentialSigmaAdder  Highres Fix Flux**  
  The PolyExponential Sigma Adder node adds the ability to manipulate curve parameters, such as adjusting the curve’s rigidity, and allows for the application of a negative poly-exponential curve to the Sigmas.
The PolyExponential Sigma Adder introduces a resolution-independent curve, ensuring a consistent adjustment to img2img processing across different resolutions. Sustitución for resolutions depending ModelSamplingFlux

- **BasicSchedulerNormalized**  
  A scheduler node with built-in denoise  normalization to ensure stable consistent sampling results across schedulers.

- **LogSigmaSamplerNode** & **LogSigmaStepSamplerNode**
  These nodes offer direct access to the internal model noise curve—the curve the model expects from sigma values during diffusion.
  LogSigmaSamplerNode enables manipulation of this curve directly, allowing users to enhance fine details, introduce natural imperfections, or soften the final image by shifting how the model interprets noise over time.
  This approach gives precise control over the model's behavior—similar in effect to techniques used by Detail Deamon or Lying Sigmas.
  These nodes are ideal for users looking to experiment with or customize the core diffusion response for artistic or technical purposes.
  
[Detailed Inforamtion on my Patreon: ](https://www.patreon.com/c/TB_LAAR)
---

## Overview

TGB’s sidecodes provide practical solutions distilled from complex development work into simple, easy-to-integrate nodes. These tools offer enhanced control over sampling dynamics, noise management, and image refinement, making them valuable assets for artists, developers, and researchers using ComfyUI and stable diffusion workflows.

Explore this repository to discover how these nodes can streamline your image generation pipeline and help push the boundaries of creative and technical possibilities.

## License

Copyright (C) 2025  Tobias Laarmann
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program comes with ABSOLUTELY NO WARRANTY; 
    This is free software, and you are welcome to redistribute it
    under certain conditions;

---

## Support

For updates and detailed development insights, visit TGB’s [Patreon page](https://www.patreon.com/c/TB_LAAR)).

---

## Tags

`ComfyUI` `Stable Diffusion` `AI Art` `Model Sampling` `Noise Scheduling` `Image Refinement` `Upscaling` `Tiled Upscaler` `Patreon` `Open Source` `AI Nodes` `Flux Control`

---

*Happy creating!*

