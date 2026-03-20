import numpy as np
from ..quadratic_obj_base import QuadraticObjectiveMixin

class SupermodularQuadraticObjectiveMixin(QuadraticObjectiveMixin):

    def _build_quadratic_coeff_batch(self, theta):
        Q = super()._build_quadratic_coeff_batch(theta)
        if Q.min() < -1e-10:
            raise ValueError(
                f"Q(theta) has negative entries (min={Q.min():.4f}). "
                f"Supermodular min-cut requires theta >= 0 for quadratic covariates."
            )
        np.maximum(Q, 0, out=Q)
        return Q