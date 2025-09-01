# Medical Image Enhancement – Noise Reduction in Ultrasound Images

This project applies **Digital Image Processing (DIP)** techniques to reduce **speckle noise** in fetal ultrasound images.  
The objective is to improve diagnostic clarity while preserving important anatomical details.  



## Problem
Ultrasound images often suffer from speckle noise, which reduces visibility of structures and makes diagnosis difficult.  
The goal of this project is to apply and compare multiple denoising methods, then design a **novel hybrid pipeline** for optimal enhancement.  



## Methods Implemented
1. **Median Filtering** – Removes noise by replacing each pixel with the median of its neighborhood.  
2. **Mean (Non-local Means) Denoising** – Uses patch similarity to smooth noise without blurring edges.  
3. **Anisotropic Diffusion** – Preserves edges while smoothing homogeneous regions.  
4. **Wavelet Transform Denoising** – Decomposes the image into frequency sub-bands, applies thresholding to suppress noise.  
5. **Hybrid Pipeline (Novelty)** – Sequential application of Wavelet → Mean → Median for maximum noise reduction and detail preservation.  



## Tech Stack
- Python  
- OpenCV, NumPy, Matplotlib  
- PyWavelets, scikit-image  



## Sample Results

**Original vs Noisy vs Enhanced (Hybrid)**

| Original | Noisy | Median | Anisotropic | Wavelet | Hybrid |
|----------|-------|--------|-------------|---------|--------|
| ![Original](assets/original.png) | ![Noisy](assets/noisy.png) | ![Median](assets/median.png) | ![Anisotropic](assets/anisotropic.png) | ![Wavelet](assets/wavelet.png) | ![Hybrid](assets/hybrid.png) |



## Quantitative Results

| Method                | PSNR ↑ | SSIM ↑ |
|------------------------|--------|--------|
| Median Filter          | 28.5   | 0.71   |
| Mean Denoising         | 29.0   | 0.72   |
| Anisotropic Diffusion  | 29.7   | 0.76   |
| Wavelet Transform      | 30.1   | 0.74   |
| Hybrid Pipeline        | **32.4** | **0.81** |


