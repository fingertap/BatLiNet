import torch
import numpy as np

from scipy.interpolate import interp1d


from src.data import BatteryData
from src.builders import FEATURE_EXTRACTORS
from src.feature.base import BaseFeatureExtractor


@FEATURE_EXTRACTORS.register()
class BatLiNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self,
                 interp_dim: int = 1000,
                 diff_base: int = None,
                 feature_to_drop: list = None,
                 cycle_to_drop: list = None,
                 smooth_features: bool = False,
                 smooth_window_size: int = 201,
                 smooth_device: str = 'cuda:0',
                 min_cycle_index: int = 0,
                 max_cycle_index: int = 99,
                 max_capacity: float = 1.2):
        """Build multi-facted feature for deep battery degradation prediction.

        Args:
            interp_dim (int, optional): Interpolation dimensionality. Defaults
                to 1000.
            diff_base (int, optional): The index of the cycle to be subtracted.
                Defaults to None.
            feature_to_drop (list, optional): Drop some of the features
                according to the index. Defaults to None.
            cycle_to_drop (list, optional): Drop some of the cycles according
                to the index. Defaults to None.
            smooth_features (bool, optional): Whether to smooth the features
                using Hampel filter. Defaults to True.
            smooth_window_size (int, optional): The window size of Hampel
                filter smoothing. Needs to be odd. Defaults to 201.
            smooth_device (str, optional): The device to accelerate smoothing.
                Defaults to cuda:0.
            min_cycle_index (int, optional): The start cycle index (inclusive)
                for feature extraction. Defaults to 0.
            max_cycle_index (int, optional): The end cycle index (inclusive)
                for feature extraction. Defaults to 99.
        """
        self.interp_dim = interp_dim
        self.diff_base = diff_base
        self.smooth_features = smooth_features
        self.smooth_window = smooth_window_size
        self.smooth_device = smooth_device
        self.min_cycle_index = min_cycle_index
        self.max_cycle_index = max_cycle_index
        self.max_capacity = max_capacity

        if isinstance(feature_to_drop, int):
            feature_to_drop = [feature_to_drop]
        self.feature_to_drop = feature_to_drop
        if isinstance(cycle_to_drop, int):
            cycle_to_drop = [cycle_to_drop]
        self.cycle_to_drop = cycle_to_drop or []

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        eps = 1e-3
        feature = []
        for cycle_indx, cycle_data in enumerate(cell_data.cycle_data):
            if cycle_indx < self.min_cycle_index:
                continue
            if cycle_indx > self.max_cycle_index:
                break
            if cycle_indx in self.cycle_to_drop:
                feature.append(torch.zeros(6, self.interp_dim))
                continue

            I = np.array(cycle_data.current_in_A)  # noqa
            V = np.array(cycle_data.voltage_in_V)
            Qc = np.array(cycle_data.charge_capacity_in_Ah) \
                / cell_data.nominal_capacity_in_Ah
            Qd = np.array(cycle_data.discharge_capacity_in_Ah) \
                / cell_data.nominal_capacity_in_Ah

            charge_mask, discharge_mask = I > 0.1, I < -0.1
            Qc, Qd = Qc[charge_mask], Qd[discharge_mask]
            Ic, Id = I[charge_mask], I[discharge_mask]
            Vc, Vd = V[charge_mask], V[discharge_mask]
            # V(Qc), V(Qd), I(Qc), I(Qd)
            cycle_feature = [
                interpolate(
                    Qc, Vc, self.interp_dim, 'charge', xe=self.max_capacity),
                interpolate(
                    Qd, Vd, self.interp_dim, 'discharge', xe=self.max_capacity),
                interpolate(Qc, Ic, self.interp_dim, xe=self.max_capacity),
                interpolate(Qd, Id, self.interp_dim, xe=self.max_capacity),
            ]
            # delta_V(Q)
            cycle_feature.append(
                cycle_feature[0] - cycle_feature[1][::-1]
            )
            # R(Q)
            cycle_feature.append(
                (cycle_feature[0] - cycle_feature[1][::-1])
                / (cycle_feature[2] - cycle_feature[3][::-1] + eps)
            )
            feature.append(np.stack(cycle_feature))

        feature = torch.from_numpy(np.stack(feature))

        if self.diff_base is not None:
            feature -= feature[[self.diff_base]]

        if self.smooth_features:
            feature = torch.stack([
                hampel_smooth(cycle, self.smooth_window, self.smooth_device)
                for cycle in feature
            ])

        if self.feature_to_drop is not None:
            to_keep = [x for x in range(feature.shape[1])
                       if x not in self.feature_to_drop]
            feature = feature[:, to_keep]

        feature = feature.transpose(1, 0)

        # Drop NaN
        feature[feature != feature] = 0.

        return feature


def interpolate(x, y, interp_dims, fill_type='', xs=0, xe=1.2):
    if len(x) <= 2:
        return np.zeros(interp_dims)
    mask = (x >= xs) & (x <= xe)
    x, y = x[mask], y[mask]
    if fill_type == 'charge':
        fill_values = (y.min(), y.max())
    elif fill_type == 'discharge':
        fill_values = (y.max(), y.min())
    else:
        fill_values = 0.
    func = interp1d(
        x, y,
        kind='linear',
        bounds_error=False,
        fill_value=fill_values)
    return func(np.linspace(xs, xe, interp_dims))


def rollingOps1d(x, func, window_size=101):
    processed = func(x.unfold(-1, window_size, 1))
    L, l = x.size(-1), processed.size(-1)  # noqa
    left = (L - l) // 2
    right = L - l - left
    res = torch.zeros_like(x)
    res[..., left:-right] = processed
    res[..., :left] = res[..., [left]]
    res[..., -right:] = res[..., [-(right+1)]]

    return res


def med1d(x, window_size=100):
    def med(x):
        return x.median(-1)[0]
    return rollingOps1d(x, med, window_size)


def mad1d(x, window_size=100):
    def mad(x):
        med = x.median(-1)[0]
        diff = (x - med.unsqueeze(-1)).abs()
        return diff.median(-1)[0]
    return rollingOps1d(x, mad, window_size)


def _hampel_smooth(x, window_size):
    med = med1d(x, window_size)
    diff = (x - med).abs()
    sigma = 1.4826 * mad1d(x, window_size) * 3

    res = x.clone()
    res[diff > sigma] = med[diff > sigma]

    return res


def hampel_smooth(x, window_size=201, device='cuda:0'):
    # x size (*, L)
    # NOTE: x should not be too large, as the unfold will expand the memory use
    #       if x is very large (e.g. [B, N, K, L] with large B and N), you can
    #       use torch.stack([x_single for x_single in x])

    assert window_size % 2 == 1, 'Window size must be odd!'
    is_array = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        is_array = True

    original_device = x.device
    x = x.to(device)
    res = _hampel_smooth(x, window_size)
    res = res.to(original_device)

    if is_array:
        res = res.cpu().numpy()

    return res
