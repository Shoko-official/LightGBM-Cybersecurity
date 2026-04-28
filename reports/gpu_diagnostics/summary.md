# GPU Diagnostics

- Python: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- LightGBM: `4.6.0`
- CUDA_HOME: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`
- CUDA_PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`
- Binary GPU toy: `success`
- Multiclass GPU toy: `failed`

## Conclusion

- Verdict: `LightGBM GPU is available on this machine, but the multiclass GPU path is unstable.`
- Likely cause: `environment_or_lightgbm_multiclass_gpu_path`