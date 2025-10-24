"""
    SE2
    ===

    Solve the Eikonal PDE on SE(2).

    Contains three submodules for different controller types on SE(2), which
    each contain methods for solving the corresponding Eikonal PDE and computing
    geodesics:
      1. Riemannian.
      2. subRiemannian.
      3. plus.

    Moreover provides the following "top level" submodule:
      1. vesselness: compute the SE(2) vesselness of an image, which can be put
      into a cost function and subsequently into a data-driven metric. 

    Additionally, we have the following "internal" submodules:
      1. derivatives: compute various derivatives of functions on SE(2).
      2. utils
"""

# Access entire backend
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.derivatives
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.vesselness
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.costfunction
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.utils
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.Riemannian
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.subRiemannian
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.plus
