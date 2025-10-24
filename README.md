# E-NES
This repository contains the code for the paper called **"Equivariant Eikonal Neural Networks: Grid-Free, Scalable Travel-Time Prediction on Homogeneous Spaces"**.

[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![jax_badge][jax_badge_link]](https://github.com/google/jax)
[![badge](https://img.shields.io/badge/ArXiv-2406.06660-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2505.16035)

[jax_badge_link]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC



# Abstract

We introduce Equivariant Neural Eikonal Solvers, a novel framework that integrates Equivariant Neural Fields (ENFs) with Neural Eikonal Solvers. Our approach employs a single neural field where a unified shared backbone is conditioned on signal-specific latent variables – represented as point clouds in a Lie group – to model diverse Eikonal solutions. The ENF integration ensures equivariant mapping from these latent representations to the solution field, delivering three key benefits: enhanced representation efficiency through weight-sharing, robust geometric grounding, and solution steerability. This steerability allows transformations applied to the latent point cloud to induce predictable, geometrically meaningful modifications in the resulting Eikonal solution. By coupling these steerable representations with Physics-Informed Neural Networks (PINNs), our framework accurately models Eikonal travel-time solutions while generalizing to arbitrary Riemannian manifolds with regular group actions. This includes homogeneous spaces such as Euclidean, position–orientation, spherical, and hyperbolic manifolds. We validate our approach through applications in seismic travel-time modeling of 2D, 3D, and spherical benchmark datasets. Experimental results demonstrate superior performance, scalability, adaptability, and user controllability compared to existing Neural Operator-based Eikonal solver methods.

# Requirements

To install the requirements, we use conda. We recommend creating a new environment for the project.
```
conda create -n eikonal-enf-jax python=3.10
conda activate eikonal-enf-jax
```

Install the relevant dependencies.
```
pip install -r requirements.txt
conda install hfm=0.2.13 -c agd-lbr
conda install -c conda-forge cupy=13.3 cuda-version=12.4
```

# Experiments
To reproduce the experiments you need to run the following commands precomposed by:
```
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 && export PYTHONPATH="."
```

## OpenFWI 2D fitting autodecoding:
Modify on the config file ```open_fwi_2d.yaml``` the data set that you would want to train and then run
```
python experiments/fitting/ad_fit.py --space euclidean --experiment open_fwi_2d
```

## OpenFWI 2D fitting meta:
Modify on the config file ```open_fwi_2d_meta.yaml``` the data set that you would want to train and then run
```
python experiments/fitting/meta_ad_fit.py --space euclidean --experiment open_fwi_2d
```

## OpenFWI 2D fitting pretrained meta:
Modify on the config file ```open_fwi_2d_mix.yaml``` the data set that you would want to train, add the wandb identifyer of the autodecoding run that you want to use as initialization, and then run
```
python experiments/fitting/meta_ad_fit.py --space euclidean --experiment open_fwi_2d
```

## OpenFWI 2D ablation of autodecoding steps:
Add in the python script the wandb ids of the runs that you want to set as the ablation, and then run the script
```
python experiments/fitting/ablation_auto_steps.py
```

## OpenFWI 2D ablation equivariance:
Modify on the config file ```open_fwi_2d.yaml``` the data set that you would want to train, and set
```
group: Sym-NoEquiv
dim_orientation: 0
```
then run 
```
python experiments/fitting/ad_fit.py --space euclidean --experiment open_fwi_2d
```

## OpenFWI 2D fitting Functa model:
Modify on the config file ```open_fwi_2d.yaml``` the data set that you would want to train and then run
```
python experiments/fitting/ad_fit_functa.py --space euclidean --experiment open_fwi_2d
```

## OpenFWI 3D fitting autodecoding:
Modify on the config file ```open_fwi_3d.yaml``` the data set that you would want to train (Style-b in the paper) and then run
```
python experiments/fitting/ad_fit.py --space euclidean --experiment open_fwi_3d
```

## OpenFWI 3D fitting pretrained meta:
Modify on the config file ```open_fwi_2d_mix.yaml``` the data set that you would want to train, add the wandb identifyer of the autodecoding run that you want to use as initialization, and then run
```
python experiments/fitting/mix_ad_fit.py --space euclidean --experiment open_fwi_3d
```

## OpenFWI 3D ablation:
Add in the python script the wandb ids of the runs that you want to set as the ablation, and then run the script
```
python experiments/fitting/ablation_3d.py
```

## Spherical fitting
Run the following scripts for the corresponding data sets

```
python experiments/fitting/ad_fit.py --space spherical --experiment open_fwi
```

```
python experiments/fitting/ad_fit.py --space spherical --experiment gauss
```

```
python experiments/fitting/ad_fit.py --space spherical --experiment homogenous
```

## Spherical Gradient integration

Add in the python script the wandb identifyer, the start and goal coordinates, the id of the velocity field, and the stepsize of the gradient decesnt. Then run the following script
```
python experiments/downstream/gradient_int_sphere.py
```


# Citation

If you find this code useful, please consider citing our paper:
```
@article{garcia2025equivariant,
  title={Equivariant Eikonal Neural Networks: Grid-Free, Scalable Travel-Time Prediction on Homogeneous Spaces},
  author={Garc{\'\i}a-Castellanos, Alejandro and Wessels, David R and Berg, Nicky J and Duits, Remco and Pelt, Dani{\"e}l M and Bekkers, Erik J},
  journal={arXiv preprint arXiv:2505.16035},
  year={2025}
}
```
