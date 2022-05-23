"""spearmanr for ranking."""

import numpy as np
from scipy.stats import spearmanr

from matchzoo.engine.base_metric import RankingMetric


class Spearmanr(RankingMetric):
    """spearmanr metric."""

    ALIAS = 'Spearmanr'

    def __init__(self):
        """:class:`Spearmanr` constructor."""

    def __repr__(self) -> str:
        """:return: format string representation of the metric."""
        return f"{self.ALIAS}"

    def __call__(self, y_true: np.array, y_pred: np.array):
        """
        Calculate spearmanr.

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        """
        return spearmanr(y_true, y_pred).correlation
