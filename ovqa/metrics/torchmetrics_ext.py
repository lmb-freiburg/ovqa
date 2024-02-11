from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.data import allclose


class MetricExt(Metric):
    """
    Other hints:
        compute() should return a single float
    """

    def compute_per_datapoint(self, return_dict=False):
        """
        Override to compute per-datapoint metric instead of scalar.

        Returns:
            metric shape (n_datapoints, ), possible other values
            or dictionary if return_dict is set.
        """
        raise NotImplementedError

    def format(self, value: float):
        """Override to format the metric value for logging."""
        return f"{value:.2%}"


# noinspection PyProtectedMember
class MetricCollectionExt(MetricCollection):
    """
    MetricCollection has problems with string states, this fixes it.
    """

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        # empty state
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False

        if metric1._defaults.keys() != metric2._defaults.keys():
            return False

        for key in metric1._defaults.keys():
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)

            if type(state1) != type(state2):
                return False

            if isinstance(state1, Tensor) and isinstance(state2, Tensor):
                return state1.shape == state2.shape and allclose(state1, state2)

            if isinstance(state1, list) and isinstance(state2, list):
                # BEGIN CHANGE
                if len(state1) != len(state2):
                    return False
                if len(state1) == 0:
                    return True
                if isinstance(state1[0], str):
                    return state1 == state2
                # END CHANGE
                return all(
                    s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2)
                )

        return True
