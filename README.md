Functional Imaging Denoising Repository

This repository hosts a comprehensive suite of denoising algorithms tailored for functional imaging, with a special emphasis on voltage imaging data. The implemented methods span tensor decomposition, self-supervised denoising, and hybrid pipelines designed to separate neural signals from complex, structured noise.

Implementations Overview
1. Tucker Decomposition with Deep Network Priors
A neural-regularized implementation of Tucker decomposition is provided, where each factor (spatial, temporal, and depth) is constrained using deep prior networks. The architecture adopts !D convolutional UNet priors for spatial and temporal components, enabling expressive yet low-rank reconstructions.

Inspired by: [DeepTensor (Petersen et al., NeurIPS 2022)], and adapted for high-dimensional microscopy data.

Captures global spatiotemporal structure in noisy videos.

Enhances interpretability through separable factor learning.

2. Comparative Analysis: Tucker vs Penalized Matrix Decomposition (PMD)
Side-by-side comparisons demonstrate:

Tucker’s superior ability to capture motion dynamics and subtle signal propagation in neuronal tissue.

PMD’s effectiveness at isolating low-rank global background but its limitations in representing fast-varying, localized activity.

Based on metrics such as SNR gain, residual energy concentration, and visual fidelity across datasets.

3. Zero-Shot Noise2Noise for Voltage Imaging
A zero-shot self-supervised denoising framework modeled after Noise2Noise is implemented:

Learns denoising directly from the noisy voltage imaging stack by leveraging unpaired frames.

No clean ground truth required.

Particularly effective in datasets where trial-averaged consistency exists.

Validated on synthetic and real voltage imaging sequences with known stimulation epochs.

4. Hybrid Tucker + SUPPORT Pipeline
Integrates the SUPPORT method (Nature Methods 2023) as a post-processing step on the Tucker-denoised residual:

Tucker handles low-rank background modeling.

SUPPORT extracts sparse spike-like events missed in Tucker’s smooth approximation.

The combination yields better temporal sharpness and spatial accuracy in capturing spiking events.

5. Power Spectrum Analysis (Radial + 1D Average)
We perform Fourier-based frequency analysis on denoised outputs to characterize differences between:

Signal vs DC/background separation.

PMD vs Tucker reconstructions.

Metrics include:

Radial power spectrum to evaluate isotropy and resolution recovery.

1D average power to assess spectral fidelity at specific frequency bands.

Reveals how PMD suppresses high-frequency signal, while Tucker recovers more neural activity within mid-frequency ranges.

6. Ongoing Work: Unified Tucker-Based Factorization into Signal + Global + Local Background
A novel model is under development which:

Decomposes the noisy video into three components:

Signal (sparse, structured dynamics)

Global background (low-rank, temporally constant)

Local background (smooth, spatially adaptive)

Incorporates weighted priors from signal-background correlation maps to guide the factorization.

Learns in a single optimization loop, reducing the need for multiple post hoc denoising steps.
