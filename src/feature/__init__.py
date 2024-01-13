"""Extract the features from a list of `BatteryData` and output a PyTorch `Dataset` object."""  # noqa
from .multifacet_capacity_matrix import \
    MultifacetedCapacityMatrixRULFeatureExtractor
from .full_model import FullModelFeatureExtractor
from .variance_model import VarianceModelFeatureExtractor
from .discharge_model import DischargeModelFeatureExtractor
from .voltage_capacity_matrix import VoltageCapacityMatrixFeatureExtractor
