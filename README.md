# Acoustic Reliefs 

Code release for *Acoustic Reliefs*; Siggraph Asia; Jeremy Chew, Michal Piovarči, Kangrui Xue, Doug James, Bernd Bickel

**Create acoustic diffusers with custom images!**

![Teaser](./submission/images/teaser.png)

## Useful Links

Learn more about this project on our [project page](https://cdl.ethz.ch/publications/acoustic-reliefs/).

Read our paper [here (author version)](./submission/paper/Acoustic_Reliefs.pdf) or [here (publisher version)]().

Get the *Acoustic Reliefs* meshes here:
- [0.6m](./submission/acoustic_reliefs/0.6/)
- [0.9m](./submission/acoustic_reliefs/0.9/)

## Abstract

We present a framework to optimize and generate *Acoustic Reliefs*: Acoustic diffusers that not only perform well acoustically in scattering sound uniformly in all directions, but are also visually interesting and can approximate user-provided images. To this end, we develop a differentiable acoustics simulator based on the boundary element method, and integrate it with a differentiable renderer coupled with a vision model to jointly optimize for acoustics, appearance, and fabrication constraints at the same time. We generate various examples and fabricate two room-scale reliefs. The result is a validated simulation and optimization scheme for generating acoustic reliefs whose appearances can be guided by a provided image.

## Examples

![Rep](./submission/images/rep_image.png)

### 0.6m Diffusers
![60](./submission/images/60.png)

### 0.9m Diffusers
![90](./submission/images/90.png)

### Office Diffusers
![office](./submission/images/office.png)

## Create Your Own

### 1. Installation

```
# Clone the repo
git clone --recursive git@github.com:mickey1356/acoustic_reliefs.git && cd acoustic_reliefs

# Install the conda environment
conda env create -f environment.yml
conda activate reliefs

# Install the C++ acoustic BEM solver
cd acoustics3d_cpp
pip install .
cd ..
```

### 2. Configuration

Take a look at the base config file in `configs/base.toml`. Most of the options should be self-explanatory, but we elaborate on several configuration options:

`[diffbem]`

Controls the hierarchical matrix parameters, as well as the overall simulation setup for a diffuser.

Set `recompute_matrices` to `true` if you don't have enough space to store the hierarchical matrix for both the forward and backward pass.

`[dimensions] - subdiv_top`

How many additional subdivisions to make for the top face. Setting a higher number allows for a greater heightmap resolution, but comes at a significant cost to runtime.

`[image] - cam_rad`

We want to set this to a value such that the relief (when rendered from the top-down) takes up as much space in the image as possible. A distance of 1 works well for 0.6m reliefs, and 1.5 works well for 0.9m reliefs.

`[optimization] - guidance_type`

Currently, only `acoustics_only`, `image`, and `image_only` are supported. `text` and `text_only` might be implemented in a future work, using (for example, the Score Distillation Sampling loss).

### 3. Running

The full pipeline consists of three stages.

**Stage 1**

The first stage runs a coarser version of the optimization:

`python acoustics_opy.py <path_to_config_file>`

**Stage 2**

The second stage improves the appearance while maintaining the relief's acoustical qualities:

`python hres_optim.py <folder_to_stage1_results>`

Please make sure that the config file (`*.toml`) exists within the above folder.

**(Optional) Stage 3**

Finally, the third stage runs the acoustics solver to compute the diffusion coefficients, writing them into a CSV file located at `outputs/coeffs.csv`. You would need to manually edit the Python script, as this was mainly used for internal purposes.

### Other Issues
If you have any issues or questions, please submit an [issue](https://github.com/mickey1356/acoustic_reliefs/issues). I will be happy to help!

### Citation

```
Not available yet.
```