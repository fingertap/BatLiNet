import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from src.builders import MODELS
from src.data.databundle import DataBundle, Dataset
from src.models.rul_predictors.cnn import ConvModule

from ..nn_model import NNModel


class DiffDataset(Dataset):
    def __init__(self, cycle_diff_feature, raw_feature, label):
        self.feature = cycle_diff_feature
        self.raw_feature = raw_feature
        self.label = label

    def __getitem__(self, indx):
        return {
            'feature': self.feature[indx],
            'label': self.label[indx],
            'raw_feature': self.raw_feature[indx]
        }


@torch.no_grad()
def smoothing(feature):
    med = feature.median(-1)[0].unsqueeze(-1).expand(*feature.shape)
    med_diff = (feature - med).abs()
    med_diff_std = med_diff.std(-1, keepdim=True).expand(*feature.shape)
    mask = med_diff > med_diff_std * 3
    feature[mask] = 0.
    return feature


@MODELS.register()
class BatLiNetRULPredictor(NNModel):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 alpha: float = 0.5,
                 kernel_size: int = 3,
                 diff_base: int = 10,
                 train_support_size: int = None,
                 test_support_size: int = None,
                 gradient_accumulation_steps: int = 1,
                 support_size: int = 1,
                 lr: float = 1e-3,
                 act_fn: str = 'relu',
                 filter_cycles: bool = True,
                 features_to_drop: list = None,
                 cycles_to_drop: list = None,
                 return_pointwise_predictions: bool = False,
                 seed: int = 0,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_height < kernel_size[0]:
            kernel_size = (input_height, kernel_size[1])
        if input_width < kernel_size[1]:
            kernel_size = (kernel_size[0], input_width)

        self.alpha = alpha
        self.channels = channels
        self.diff_base = diff_base
        self.train_support_size = train_support_size or support_size
        self.test_support_size = test_support_size or support_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.filter_cycles = filter_cycles
        if isinstance(features_to_drop, int):
            features_to_drop = [features_to_drop]
        self.features_to_drop = features_to_drop
        if isinstance(cycles_to_drop, int):
            cycles_to_drop = [cycles_to_drop]
        self.cycles_to_drop = cycles_to_drop
        self.return_pointwise_predictions = return_pointwise_predictions

        self.ori_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        self.sup_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        # Shared regressor without bias
        self.fc = nn.Linear(channels, 1, bias=False)
        self.lr = lr
        self.seed = seed

    def forward(self,
               feature: torch.Tensor,
               label: torch.Tensor,
               support_feature: torch.Tensor,
               support_label: torch.Tensor,
               return_loss: bool = False):
        B, S, C, H, W = support_feature.size()

        x_ori = self.ori_module(feature)
        x_sup = self.sup_module(support_feature.view(-1, C, H, W))

        y_ori = self.fc(x_ori.view(B, self.channels)).view(-1)
        y_sup = self.fc(x_sup.view(B, S, self.channels)).view(B, S)
        y_sup += support_label.view(B, S)

        if self.return_pointwise_predictions:
            return y_ori, y_sup

        if self.training:
            y_sup = y_sup.mean(1).view(-1)
        else:
            # We use median aggregation to minimize the influence of outliers
            y_sup = y_sup.median(1)[0].view(-1)

        if return_loss:
            loss = sum([
                (1. - self.alpha) * mse(y_ori, label),
                self.alpha * mse(y_sup, label)
            ])
            return loss

        return (1. - self.alpha) * y_ori + self.alpha * y_sup

    def fit(self, dataset: DataBundle, timestamp: str):
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        # Build a cycle diff dataset
        train_dataset = self.build_cycle_diff_dataset(dataset.train_data)
        ori_loader = DataLoader(
            train_dataset, self.train_batch_size, shuffle=False)

        latest = None
        for epoch in tqdm(range(self.train_epochs), desc='Training'):
            self.train()

            for indx, data_batch in enumerate(ori_loader):
                x, y, raw_x = data_batch.values()
                sup_x, sup_y = self.get_support_set(
                    raw_x, dataset.train_data.feature, dataset.train_data.label)
                loss = self.forward(x, y, sup_x, sup_y, return_loss=True)
                loss.backward()

                if (
                    indx == len(ori_loader) - 1
                    or (indx + 1) % self.grad_accum_steps == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()

            if (
                self.workspace is not None
                and self.checkpoint_freq is not None
                and (epoch + 1) % self.checkpoint_freq == 0
            ):
                filename = self.workspace / f'{timestamp}_seed_{self.seed}_epoch_{epoch+1}.ckpt'
                self.dump_checkpoint(filename)
                latest = filename

            if (epoch + 1) % self.evaluate_freq == 0:
                del loss, sup_x, sup_y, x, y
                pred = self.predict(dataset)
                score = dataset.evaluate(pred, 'RMSE')
                message = f'[{epoch+1}/{self.train_epochs}] RMSE {score:.2f}'
                print(message, flush=True)
                del pred

        # Create symlink latest
        if latest is not None and self.workspace is not None:
            self.link_latest_checkpoint(latest)

    @torch.no_grad()
    def predict(self, dataset: DataBundle) -> torch.Tensor:
        self.eval()
        # Build a cycle diff dataset
        test_dataset = self.build_cycle_diff_dataset(dataset.test_data)
        ori_loader = DataLoader(
            test_dataset, self.test_batch_size, shuffle=False)
        predictions = []
        for indx, data_batch in enumerate(ori_loader):
            x, y, raw_x = data_batch.values()
            sup_x, sup_y = self.get_support_set(
                raw_x, dataset.train_data.feature, dataset.train_data.label)
            predictions.append(self.forward(x, y, sup_x, sup_y))
        if self.return_pointwise_predictions:
            predictions = (
                torch.cat([x[0] for x in predictions]),
                torch.cat([x[1] for x in predictions]),
            )
        else:
            predictions = torch.cat(predictions)
        return predictions

    @torch.no_grad()
    def build_cycle_diff_dataset(self, dataset: Dataset):
        feature = dataset.feature - dataset.feature[:, :, [self.diff_base]]
        raw_feature = dataset.feature
        if self.features_to_drop is not None:
            mask = [x for x in range(feature.size(1))
                    if x not in self.features_to_drop]
            feature = feature[:, mask].contiguous()
            raw_feature = raw_feature[:, mask].contiguous()
        if self.cycles_to_drop is not None:
            feature[:, :, self.cycles_to_drop] = 0.
            raw_feature[:, :, self.cycles_to_drop] = 0.
        feature = self._clean_feature(feature)
        raw_feature = self._filter_cycles(raw_feature)
        return DiffDataset(feature, raw_feature, dataset.label)

    @torch.no_grad()
    def get_support_set(self, x, sup_feat, sup_label):
        if self.features_to_drop is not None:
            mask = [i for i in range(sup_feat.size(1))
                    if i not in self.features_to_drop]
            sup_feat = sup_feat[:, mask].contiguous()
        if self.cycles_to_drop is not None:
            sup_feat[:, :, :, self.cycles_to_drop] = 0.
        if self.training:
            size = (len(x) * self.train_support_size,)
        else:
            size = (len(x) * self.test_support_size,)
        indx = torch.randint(len(sup_feat), size, device=x.device)
        B, C, H, W = x.size()
        feature = x.unsqueeze(1) - sup_feat[indx].view(B, -1, C, H, W)
        label = sup_label[indx].view(B, -1)
        feature = self._clean_feature(feature)
        return feature, label

    def _clean_feature(self, feature):
        num = 50
        feature[..., :num] = smoothing(feature[..., :num])
        feature[..., -num:] = smoothing(feature[..., -num:])
        feature = remove_glitches(feature)
        # Filter problematic cycles using Hampel filter
        feature = self._filter_cycles(feature)
        return feature

    def _filter_cycles(self, feature):
        if not self.filter_cycles:
            return feature
        feature = feature.clone()

        # Filter the cycles with its max value too large
        max_val = feature.abs().amax(-1)
        max_val_med = max_val.median(-1, keepdim=True)[0]
        max_val_diff = (max_val - max_val_med).abs()
        mask = max_val_diff > max_val_diff.std(-1, keepdim=True) * 5

        # Filter the cycles with its mean deviating from other cycles
        mean_val = feature.mean(-1)
        mean_val_med = mean_val.median(-1, keepdim=True)[0]
        mean_val_diff = (mean_val - mean_val_med).abs()
        mask |= mean_val_diff > mean_val_diff.std(-1, keepdim=True) * 5

        # Fill with zero
        feature[mask] = 0.

        return feature


def _remove_glitches(x, width, threshold):
    left_element = torch.roll(x, shifts=1, dims=-1)
    right_element = torch.roll(x, shifts=-1, dims=-1)
    diff_with_left_element = (left_element - x).abs()
    diff_with_right_element = (right_element - x).abs()

    # diff_with_left_element[..., 0] = 0.
    # diff_with_right_element[..., -1] = 0.

    ths = diff_with_left_element.std(-1, keepdim=True) * threshold
    non_smooth_on_left = diff_with_left_element > ths
    ths = diff_with_right_element.std(-1, keepdim=True) * threshold
    non_smooth_on_right = diff_with_right_element > ths
    for _ in range(width):
        non_smooth_on_left |= torch.roll(
            non_smooth_on_left, shifts=1, dims=-1)
        non_smooth_on_right |= torch.roll(
            non_smooth_on_right, shifts=-1, dims=-1)
    to_smooth = non_smooth_on_left & non_smooth_on_right
    x[to_smooth] = 0.
    return x


def remove_glitches(data, width=25, threshold=3):
    shape = data.shape
    data = data.view(-1, *shape[-3:])
    for i in range(len(data)):
        data[i] = _remove_glitches(data[i], width, threshold)
    data = data.view(shape)
    return data


def build_module(
    in_channels, channels, input_height, input_width, kernel_size, act_fn
) -> nn.Module:
    encoder = ConvModule(in_channels, channels, kernel_size, act_fn)
    H, W = encoder.output_shape(input_height, input_width)
    proj = nn.Conv2d(channels, channels, (H, W))
    return nn.Sequential(encoder, proj, nn.ReLU())


def mse(pred, label):
    return ((pred.view(-1) - label.view(-1)) ** 2).mean()
