# E-NES
This repository contains the code for the paper called **"Equivariant Eikonal Neural Networks: Grid-Free, Scalable Travel-Time Prediction on Homogeneous Spaces"**.

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