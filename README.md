<div align="center">

<img src="assets/logo.svg" alt="Logo" style="width: 75%; height: auto;">

[![Blog](https://img.shields.io/badge/blog-blue)](https://rectifiedflow.github.io)
[![Book](https://img.shields.io/badge/book-blue)](https://arxiv.org/pdf/2209.03003.pdf)

______________________________________________________________________

</div>


**RectifiedFlow** is a repository that presents a *unified* and *minimal* codebase for implementing various diffusion and flow models, including [Rectified Flow](https://arxiv.org/pdf/2209.03003.pdf), [DDPM](https://arxiv.org/abs/2006.11239.pdf), [DDIM](https://arxiv.org/abs/2010.02502.pdf), etc.

---


# Contents
- [Installation](#installtion)
- [Training](#training)
  - [CIFAR-10](#cifar-10)
  - [ImageNet](#imagenet)
- [Sampling](#sampling)
- [License](#license)


# Installation

Please run the following commands in the given order to install the dependency.


```
conda create -n rf python=3.10
conda activate rf
git clone https://github.com/lqiang67/rectified-flow.git
cd rectified-flow
pip install -r requirements.txt
```

Then install the `rectified-flow` package:

```
pip install -e .
```

# Training

## CIFAR-10


```shell
todo
```

## ImageNet


```shell
todo
```


# Sampling

```shell
todo
```

# Citation
If you find this repository useful for your research, please consider citing

```bibtex
@misc{lq2024rectifiedflow,
  author = {Qiang Liu, Runlong Liao, Bo Liu, Xixi Hu},
  title = {PyTorch RectifiedFlow},
  year = {2024},
  url = {https://github.com/lqiang67/rectified-flow}
}
```

# License

| Component | License                                                      |
| --------- | ------------------------------------------------------------ |
| Codebase  | [MIT License](LICENSE)                                       |

