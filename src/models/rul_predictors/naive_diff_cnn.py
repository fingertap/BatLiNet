import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from src.builders import MODELS
from src.data.databundle import DataBundle, Dataset
from src.models.rul_predictors.cnn import ConvModule
from src.feature.batlinet import hampel_smooth

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
def smoothing(tensor):
    shape = tensor.shape[:-2]
    H, W = tensor.shape[-2:]
    return torch.stack([
        hampel_smooth(x, 51, tensor.device)
        for x in tensor.view(-1, H, W)
    ]).view(*shape, H, W)


@torch.no_grad()
def diff_smooth(tensor, step=200):
    W = tensor.size(-1)
    mean = tensor.abs().mean(-1, keepdim=True)
    std = tensor.abs().std(-1, keepdim=True)
    tensor[(tensor - mean).abs() > 3 * std] = 0.
    for i in range(W // step):
        data = tensor[..., i*step:(i+1)*step]
        med_diff = (data - data.median(-1)[0].unsqueeze(-1)).abs()
        ths = med_diff.median(-1)[0].unsqueeze(-1) * 3
        data[med_diff > ths] = 0.
    # mean = tensor.abs().mean(-1, keepdim=True)
    # std = tensor.abs().std(-1, keepdim=True)
    # tensor[(tensor - mean).abs() > 3 * std] = 0.
    return tensor


class CNNEncoder(nn.Module):
    def __init__(self, channels, H, W):
        nn.Module.__init__(self)
        self.proj = nn.Conv2d(channels, channels, (H, W))

    def forward(self, x):
        x = self.proj(x)
        x = x * (x < 1.5).float()
        x = torch.relu(x)
        return x


def build_module(
    in_channels, channels, input_height, input_width, kernel_size, act_fn
) -> nn.Module:
    encoder = ConvModule(in_channels, channels, kernel_size, act_fn)
    H, W = encoder.output_shape(input_height, input_width)
    proj = nn.Conv2d(channels, channels, (H, W))
    return nn.Sequential(encoder, proj, nn.ReLU())


def mse(pred, label):
    return ((pred.view(-1) - label.view(-1)) ** 2).mean()


@MODELS.register()
class NaiveDifferenceCNNRULPredictor(NNModel):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 alpha: float = 0.5,
                 kernel_size: int = 3,
                 diff_base: int = 9,
                 train_support_size: int = None,
                 test_support_size: int = None,
                 gradient_accumulation_steps: int = 1,
                 support_size: int = 1,
                 cycle_to_drop: int = None,
                 filter_cycles: bool = True,
                 loss_term: list = None,
                 lr: float = 1e-3,
                 act_fn: str = 'relu',
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
        self.loss_term = loss_term
        self.train_support_size = train_support_size or support_size
        self.test_support_size = test_support_size or support_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.filter_cycles = filter_cycles
        self.cycle_to_drop = cycle_to_drop

        self.ori_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        self.sup_module = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        self.ori_fc = nn.Linear(channels, 1)
        self.sup_fc = nn.Linear(channels, 1)

        self.lr = lr

    def forward(self,
               feature: torch.Tensor,
               label: torch.Tensor,
               support_feature: torch.Tensor,
               support_label: torch.Tensor,
               return_loss: bool = False):
        B, S, C, H, W = support_feature.size()
        if self.cycle_to_drop is not None:
            support_feature[:, :, :, self.cycle_to_drop] = 0.
            feature[:, :, self.cycle_to_drop] = 0.

        x_ori = self.ori_module(feature)
        x_sup = self.sup_module(support_feature.view(-1, C, H, W))

        x_ori = x_ori.view(B, self.channels)
        x_sup = x_sup.view(B, S, self.channels)
        y_ori = self.ori_fc(x_ori).view(-1)
        y_sup = self.sup_fc(x_sup).view(B, S) + support_label.view(B, S)
        if self.training:
            y_sup = y_sup.mean(1)
        else:
            y_sup = y_sup.median(1)[0]

        if return_loss:
            return sum([
                (1 - self.alpha) * mse(y_ori, label),
                self.alpha * mse(y_sup, label),
            ])

        pred = (1. - self.alpha) * y_ori + self.alpha * y_sup
        return pred


    @torch.no_grad()
    def build_cycle_diff_dataset(self, dataset: Dataset):
        feature = dataset.feature - dataset.feature[:, :, [self.diff_base]]
        feature = smoothing(feature)
        # Filter problematic cycles using Hampel filter
        if self.filter_cycles:
            feature = feature.permute(0, 2, 1, 3).contiguous()
            cycle_mean = feature.abs().mean(-1).amax(-1)
            med = cycle_mean.median(-1)[0]
            med_diff = (cycle_mean - med.unsqueeze(-1)).abs()
            med_diff_med = med_diff.median(-1)[0]
            cycles_to_drop = med_diff > med_diff_med.unsqueeze(-1) * 3
            feature[cycles_to_drop] = 0.
            feature = feature.permute(0, 2, 1, 3).contiguous()
        return DiffDataset(feature, dataset.feature, dataset.label)

    @torch.no_grad()
    def get_support_set(self, x, sup_feat, sup_label):
        if self.training:
            size = (len(x) * self.train_support_size,)
        else:
            size = (len(x) * self.test_support_size,)
        indx = torch.randint(len(sup_feat), size, device=x.device)
        B, C, H, W = x.size()
        feature = x.unsqueeze(1) - sup_feat[indx].view(B, -1, C, H, W)
        label = sup_label[indx].view(B, -1)
        # Filter glitches created by mismatch
        # feature = diff_smooth(feature)
        # Filter problematic cycles
        if self.filter_cycles:
            feature = feature.permute(0, 1, 3, 2, 4).contiguous()
            cycle_mean = feature.abs().mean(-1).amax(-1)
            med = cycle_mean.median(-1)[0]
            med_diff = (cycle_mean - med.unsqueeze(-1)).abs()
            med_diff_med = med_diff.median(-1)[0]
            cycles_to_drop = med_diff > med_diff_med.unsqueeze(-1) * 3
            feature[cycles_to_drop] = 0.
            feature = feature.permute(0, 1, 3, 2, 4).contiguous()
        return feature, label

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
                filename = self.workspace / f'{timestamp}_epoch_{epoch+1}.ckpt'
                self.dump_checkpoint(filename)
                latest = filename

            if (epoch + 1) % self.evaluate_freq == 0:
                del loss, sup_x, sup_y, x, y
                pred = self.predict(dataset)
                score = dataset.evaluate(pred, 'RMSE')
                message = f'[{epoch+1}/{self.train_epochs}] RMSE {score:.2f}'
                # message += f' LR: {scheduler.get_last_lr()}'
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
        predictions = torch.cat(predictions)
        return predictions
