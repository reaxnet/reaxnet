<h1>ReaxNet</h1>

![vistors](https://visitor-badge.laobi.icu/badge?page_id=reaxnet.reaxnet&right_color=green) 
<a href='https://arxiv.org/abs/2410.13820'><img src='https://img.shields.io/badge/arXiv-2403.13820-blue'></a>

![framework](site/framework.png)

The JAX implementation of polarizable long-rang interactions integrated equivariant neural network potentials (ReaxNet).
## Installation

### Easy install
```bash
pip install git+https://github.com/reaxnet/reaxnet.git
```

### Advanced install (recommend)

For NVIDIA-GPU acceleration, you should compile the JAX library with CUDA support. Please refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html#installation) for other platforms acceleration.

```bash
pip install -U "jax[cuda12]"
pip install git+https://github.com/reaxnet/reaxnet.git
```

## Usage
Please refer to [examples](./examples/). It details how to use the ReaxNet.

## Fine-tuning the pretrained model
We provide a pretrained model, which can be used to fine-tune on your own dataset. The detailed fine-tuning process can be found in the [fine_tuning.ipynb](./examples/fine_tuning.ipynb).

## Code test environment
### Python Dependencies
- Python 3.9
- JAX 0.4.20
- JAX-MD 0.2.8
- NumPy 1.23.4
- ASE 3.23.0
- e3nn_jax 0.20.7
- flax 0.10.0
- jraph 0.0.6
- matscipy 1.0.0
- optax 0.1.8

### OS 
This package is supported for macOS and Linux. The code has tested on:
- Ubuntu 22.04.4 LTS
- MacOS 14.7

## Reference

If you use this repository, please cite the following [preprint](https://doi.org/10.48550/arXiv.2410.13820):
```bib
@article{gao2024enhancing,
  title={Enhancing universal machine learning potentials with polarizable long-range interactions},
  author={Gao, Rongzhi and Yam, ChiYung and Mao, Jianjun and Chen, Shuguang and Chen, GuanHua and Hu, Ziyang},
  journal={arXiv preprint arXiv:2410.13820},
  year={2024}
}
```
