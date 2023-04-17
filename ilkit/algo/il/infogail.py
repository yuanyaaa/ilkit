from typing import Dict, Union

import numpy as np
import torch as th
from mllogger import IntegratedLogger

from ilkit.algo.base import ILPolicy


class InfoGAIL(ILPolicy):
    """Interpretable Imitation Learning from Visual Demonstrations (InfoGAIL)
    """

    def __init__(self, cfg: Dict, logger: IntegratedLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        ...

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg
    ) -> Union[np.ndarray, th.Tensor]:
        ...

    def update(self):
        ...
