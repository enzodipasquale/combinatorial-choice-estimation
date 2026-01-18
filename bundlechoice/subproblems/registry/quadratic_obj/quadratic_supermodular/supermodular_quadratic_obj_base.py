import numpy as np
from functools import lru_cache
from ..quadratic_obj_base import QuadraticObjectiveMixin

class SupermodularQuadraticObjectiveMixin(QuadraticObjectiveMixin):

    @lru_cache(maxsize=1)
    def _get_qinfo(self, _version):
        qinfo = self.data_manager.quadratic_data_info
        ad, id = self.data_manager.local_data["id_data"], self.data_manager.local_data["item_data"]
        for Q, ax in [(ad.get('quadratic'), (1, 2)), (id.get('quadratic'), (0, 1))]:
            if Q is not None:
                assert np.all(np.diagonal(Q, axis1=ax[0], axis2=ax[1]) == 0) and np.all(Q >= 0)
        return qinfo