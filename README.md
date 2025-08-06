Functional Imaging Denoising Repository

This repository in the implementation of a denoising algorithm tailored for functional imaging. The implemented methods span tensor decomposition, self-supervised denoising, and hybrid pipelines designed to separate neural signals from complex, structured noise.

Implementations Overview

Tucker Decomposition with Deep Network Priors
A neural-regularized implementation of Tucker decomposition is provided, where each factor (spatial, temporal, and depth) is constrained using deep prior networks. The architecture adopts 1D convolutional UNet priors for spatial and temporal components, enabling expressive yet low-rank reconstructions.

Inspired by: [DeepTensor (Petersen et al., NeurIPS 2022)], and adapted for high-dimensional microscopy data. Captures global spatiotemporal structure in noisy videos. Enhances interpretability through separable factor learning.

2. Ongoing Work: Unified Tucker-Based Factorization into Signal + Global + Local Background
A novel model is under development which decomposes the noisy video into three components:

Signal (sparse, structured dynamics)

Global background (low-rank, temporally constant)

Local background (smooth, spatially adaptive)

Incorporates weighted priors from signal-background correlation maps to guide the factorization. Learns in a single optimization loop, reducing the need for multiple post hoc denoising steps.
