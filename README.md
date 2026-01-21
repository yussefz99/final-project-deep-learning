# final-project-deep-learning

This project implements a diffusion-based inpainting pipeline using **Stable Diffusion 2 Base** (`stable-diffusion-2-base`) and demonstrates **inpainting with text prompts** on **our own images and masks**.  
We start from a vanilla inpainting pipeline and then identify its issues and propose improvements, evaluated both qualitatively and quantitatively.

## 1) Repository Structure
```text
report.pdf
README.md
not_our_work.txt
code/
    data/
       images/
       masks/
    scripts/
       run_inpaint_baseline.py
   
```


## 2) Environment Setup (Technion GPU Server)

We recommend running on the Technion GPU server (Linux) with conda.

### 2.1 Create / activate environment
```bash
conda create -n inpaint310 python=3.10 -y
conda activate inpaint310
```

### 2.2 Install dependencies
```bash
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors huggingface_hub pillow opencv-python tqdm matplotlib
pip install lpips   # optional (perceptual metric)
```



## 3) Model Download / Caching (one-time)
Run once on the server:
```bash
python code/scripts/cache_model.py
```
The model is cached under ~/.cache/huggingface/ (or under the cache path defined by HF_HOME / HF_HUB_CACHE).


## 4) Running Inpainting
### 4.1 Prepare data
Place your own images and masks under:
data/images/
data/masks/

Expected format:
images: .png / .jpg
masks: .png where white = hole (to inpaint) and black = keep

### 4.2 Run baseline
```bash
python code/scripts/run_inpaint.py --config code/configs/baseline.yaml
```

### 4.3 Run improved method
```bash
python code/scripts/run_inpaint.py --config code/configs/improved.yaml
```


Outputs are saved to:
outputs/baseline/
outputs/improved/

Each result is saved as a side-by-side image:
original | mask | output





