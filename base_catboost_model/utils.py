#
# Copyright (c) 2023-now by BUSUP TECHNOLOGIES, S.L.
# This file is part of project Brainer,
# and is released under the MIT License Agreement.
# See the LICENSE file for more information.
#
"""**Utility functions and classes**."""
from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    """Class to store evaluation metrics of a model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score_loss: float
    confusion_matrix: np.ndarray
